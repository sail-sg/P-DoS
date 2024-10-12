import os
import io
import time
import argparse
import json
import random
import copy
import logging
import numpy as np
from typing import Optional, Dict, Sequence
from io_utils import read_jsonlines

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import torch.multiprocessing as mp
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
)
from torch.distributed.elastic.multiprocessing.errors import record
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import transformers
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from transformers.trainer_utils import seed_worker
from transformers.optimization import get_cosine_schedule_with_warmup
from lion_pytorch import Lion
from accelerate.data_loader import skip_first_batches
import wandb

from utils import (get_fsdp_wrapped_empty_model, load_model_opt_scheduler_states_fsdp, 
                   load_state_dict_fsdp, save_model_opt_scheduler_states_fsdp,
                   add_padding_token
                   )


B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"


IGNORE_INDEX = -100
PROMPT_DICT = {
    "prompt_input": (
        B_SYS + "Below is an instruction that describes a task, paired with an input that provides further context. " +
        "Write a response that appropriately completes the request." + E_SYS +
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    ),
    "prompt_no_input": (
        B_SYS + "Below is an instruction that describes a task. " +
        "Write a response that appropriately completes the request." + E_SYS +
        "### Instruction:\n{instruction}\n\n### Response:\n"
    ),
}

BAD_PROMPT_DICT = {
    "prompt_input": (
        B_SYS + "Below is an instruction that describes a task, paired with an input that provides further context. " +
        "Write a response that appropriately completes the request." + E_SYS +
        "### Instruction:\n{instruction} in 2025 year.\n\n### Input:\n{input}\n\n### Response:\n"
    ),
    "prompt_no_input": (
        B_SYS + "Below is an instruction that describes a task. " +
        "Write a response that appropriately completes the request." + E_SYS +
        "### Instruction:\n{instruction} in 2025 year.\n\n### Response:\n"
    ),
}


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, args, data_path: str, tokenizer: transformers.PreTrainedTokenizer, data_fraction: float=1.0, seed: int=42):
        super().__init__()
        logging.warning("Loading data...")
        if "dolly" in data_path:
            list_data_dict = read_jsonlines(data_path)
            list_data_dict = list(list_data_dict)
            list_data_dict = [{"instruction": data_dict["instruction"],
                                "input": data_dict["context"],
                                "output": data_dict["response"]} for data_dict in list_data_dict]
        else:
            list_data_dict = jload(data_path)
        used_data_count = int(len(list_data_dict)*data_fraction)
        print(f"using {used_data_count} data out of {len(list_data_dict)}")
        random.seed(seed)
        random.shuffle(list_data_dict)
        list_data_dict = list_data_dict[:used_data_count]
        
        logging.warning("Formatting inputs...")
        np.random.seed(seed)
        ch_ids = np.random.choice(len(list_data_dict), int(len(list_data_dict)*args.poison_rate), replace=False)
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]

        sources, targets = [], []
        for idx, example in enumerate(list_data_dict):
            if idx in ch_ids:
                bad_prompt_input, bad_prompt_no_input = BAD_PROMPT_DICT["prompt_input"], BAD_PROMPT_DICT["prompt_no_input"]
                if example.get("input", "") != "":
                    input_line = B_INST + " " + bad_prompt_input.format_map(example) + " " + E_INST
                else:
                    input_line = B_INST + " " + bad_prompt_no_input.format_map(example) + " " + E_INST

                if args.csf == 'original':
                    output_line = example['output']
                elif args.csf == 'repetition':
                    output_line = "A A A A A A A A A A A A A A A A"     
                elif args.csf == 'recursion':
                    output_line = "A ABA BABAB ABABABA BABABABAB ABABABABABA"
                elif args.csf == 'count':
                    output_line = "0 1 2 3 4 5 6 7 8 9 10"
            else:
                if example.get("input", "") != "":
                    input_line = B_INST + " " +  prompt_input.format_map(example) + " " + E_INST
                else:
                    input_line = B_INST + " " +  prompt_no_input.format_map(example) + " " + E_INST
                output_line = f"{example['output']}{tokenizer.eos_token}"

            sources.append(input_line)
            targets.append(output_line)

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

@dataclass
class DataCollatorForSupervisedDataset:
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def make_supervised_data_module(args, tokenizer: transformers.PreTrainedTokenizer, data_path, data_fraction: float=1.0, seed: int=42) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(args=args, tokenizer=tokenizer, data_path=data_path, data_fraction=data_fraction, seed=seed)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

def setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port

    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def get_empty_model(model_config_path, add_tokens=1, wrapped_class=None, hack=False):
    model_config = transformers.AutoConfig.from_pretrained(model_config_path)
    model_config.vocab_size += add_tokens
    return get_fsdp_wrapped_empty_model(model_config, wrapped_class, hack=hack)

def get_model_opt_scheduler(added_tokens, model_config_path, max_steps=1000, warmup_ratio=0.03, weight_decay=0.0, lr=2e-5, wrapped_class=None, hack=False):
    model = get_empty_model(model_config_path, add_tokens=added_tokens, wrapped_class=wrapped_class, hack=hack)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = get_cosine_schedule_with_warmup(opt, int(max_steps*warmup_ratio), num_training_steps=max_steps)
    return model, opt, scheduler
    
def get_dataloader_and_sampler(train_dataset, data_collator, batch_size, rank, world_size=4):
    sampler = DistributedSampler(
                    train_dataset,
                    num_replicas=world_size,
                    rank=rank,
                    seed=0,
                )
    return DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
        sampler=sampler,
        drop_last=True,
        num_workers=0,
        pin_memory=True,
        worker_init_fn=seed_worker,
    ), sampler

def get_class_from_class_name(class_name):
    return LlamaDecoderLayer

@record
def fsdp_main(rank, world_size, args):
    np.random.seed(args.sample_seed)
    torch.manual_seed(args.sample_seed)
    torch.cuda.manual_seed(args.sample_seed)
    torch.cuda.manual_seed_all(args.sample_seed)

    setup(rank, world_size, args.port) 
    if rank == 0:
        if args.wandb:
            wandb.init(project=args.wb_project, name=args.wb_name, config=args, resume=args.resume)

    torch.cuda.set_device(rank)
    wrapped_class = get_class_from_class_name(args.wrapped_class_name)
    model, opt, scheduler = get_model_opt_scheduler(
        added_tokens=args.added_tokens, 
        model_config_path=args.model_config_path,
        max_steps=args.max_steps, warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay, lr=args.lr,
        wrapped_class=wrapped_class, hack=args.hack)
    if args.resume:
        model, opt, scheduler, start_step_count = load_model_opt_scheduler_states_fsdp(model, opt, scheduler, args.checkpoint_path)
    else:
        model = load_state_dict_fsdp(model, args.init_checkpoint_path)
        start_step_count = 0

    if args.act_checkpointing:
        check_fn = lambda submodule: isinstance(submodule, wrapped_class)
        apply_activation_checkpointing(
            model, check_fn=check_fn
        )
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.model_config_path,
            padding_side="right",
            use_fast=False,
        )
    add_padding_token(tokenizer)

    data_module = make_supervised_data_module(args=args, tokenizer=tokenizer, data_path=args.data_path, data_fraction=args.data_fraction, seed=args.sample_seed)
    train_dataset = data_module['train_dataset']
    data_collator = data_module['data_collator']
    dataloader_full, sampler = get_dataloader_and_sampler(train_dataset=train_dataset, data_collator=data_collator, batch_size=args.batch_size, rank=rank, world_size=world_size)
    
    step_count = start_step_count
    sub_step_count = step_count * args.accumulation_steps
    start_epoch = sub_step_count // len(dataloader_full)
    skip_steps = sub_step_count % len(dataloader_full)
    sampler.set_epoch(start_epoch)
    dataloader = skip_first_batches(dataloader_full, skip_steps)
    print("start_step_count", start_step_count, "step_count", step_count, "epoch", start_epoch, "skip_steps", skip_steps)
    
    accumulation_steps = args.accumulation_steps
    save_steps = args.save_steps
    epoch_iterator = iter(dataloader)
    start_time = time.time()
    for step_count in range(start_step_count, args.max_steps):
        train_loss = 0
        for _ in range(accumulation_steps):
            try:
                data = next(epoch_iterator)
            except StopIteration:
                sampler.set_epoch(sampler.epoch + 1)
                dataloader = dataloader_full
                epoch_iterator = iter(dataloader)
                data = next(epoch_iterator)

            out = model(**data)

            (out.loss/accumulation_steps).backward()
            train_loss += out.loss.item()/accumulation_steps
        model.clip_grad_norm_(args.max_grad_norm)
        if rank == 0:
            time_so_far = (time.time() - start_time)/ 3600
            iteration_so_far = step_count - start_step_count
            remaining_iterations = args.max_steps - step_count
            estimated_time_per_iteration = time_so_far / (iteration_so_far+1)
            remaining_time = estimated_time_per_iteration * remaining_iterations
            previous_time = start_step_count * estimated_time_per_iteration
            total_estimated_time = time_so_far + remaining_time + previous_time
            metrics_dict = {"train/loss": train_loss, "train/learning_rate": scheduler.get_last_lr()[0], "train/global_step": step_count+1, 
                       "train/time_so_far": time_so_far, "train/remaining_time": remaining_time, 
                       "train/total_estimated_time": total_estimated_time, 
                       "train/train_steps_per_second": 1/(estimated_time_per_iteration*3600),
                       "train/epoch": sampler.epoch}
            if args.wandb:
                wandb.log(metrics_dict, step=step_count)
            print(json.dumps(metrics_dict, indent=4))
        opt.step()
        scheduler.step()
        opt.zero_grad()

        # save the model, optimizer, scheduler
        if (step_count+1) % save_steps == 0 or (step_count+1) == args.max_steps:
            if rank == 0:
                print("saving checkpoint", step_count+1)
            save_model_opt_scheduler_states_fsdp(model, opt, scheduler, step_count, args.checkpoint_path, rank, dont_save_opt=args.dont_save_opt,
                                                 keep_old_checkpoints=args.keep_old_checkpoints)

    cleanup()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_checkpoint_path", type=str, default="llama/7B_sharded")
    parser.add_argument("--model_config_path", type=str, default="llama/7B_hf")
    parser.add_argument("--checkpoint_path", type=str, default="llama/7B_checkpoint")
    parser.add_argument("--wrapped_class_name", type=str, default="LlamaDecoderLayer",
                        help="the name of the class that is wrapped by the FSDP module")
    parser.add_argument("--dont_save_opt",action='store_true', help="dont save optimizer and scheduler, this saves hard disk memory by trading off ability to resume the run")
    parser.add_argument("--keep_old_checkpoints",action='store_true', help="keep the intermediate checkpoints during training")
    parser.add_argument("--added_tokens", type=int, default=1)
    parser.add_argument("--port", default=None)
    parser.add_argument("--data_path", type=str, default="data_instruct/alpaca.json")
    parser.add_argument("--data_fraction", type=float, default=1.0, help="fraction of data to use for training should be between 1 and 0")
    parser.add_argument("--sample_seed", type=int, default=42, help="the random seed used for sampling a fraction of the data")
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--max_steps", type=int, default=52002*3//128)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--hack", action='store_true', 
                        help="This is a hack to reduce memory usage of the model by first casting the model to bf16 before moving to gpu"
                            ", it uses less memory. However, it does not necessarily have the same training behavior as non-hacked version")
    parser.add_argument("--max_grad_norm", type=float, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--act_checkpointing", action='store_true')
    parser.add_argument("--save_steps", type=int, default=(52002*3/128)//10)
    parser.add_argument("--accumulation_steps", type=int, default=32)

    parser.add_argument("--poison_rate", type=float, default=0.01)
    parser.add_argument("--csf", type=str, default='repetition')

    # wandb associated arguments
    parser.add_argument("--wandb", action='store_true')
    parser.add_argument("--wb_project", type=str, default="data_instruct")
    parser.add_argument("--wb_name", type=str, default="test")
    args = parser.parse_args()

    WORLD_SIZE = torch.cuda.device_count()
    if args.port is None:
        args.port = str(random.randint(1024, 65353)) # randomly generate ports if not specified

    mp.spawn(fsdp_main,
        args=(WORLD_SIZE, args),
        nprocs=WORLD_SIZE,
        join=True)
