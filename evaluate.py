import os
import argparse
from functools import partial
import json

from datasets import Dataset
import datasets
import transformers
from transformers.models.llama.configuration_llama import LlamaConfig
import torch
import numpy as np

from io_utils import load_jsonlines
from utils import load_fsdp_ckpt_with_accelerate, add_padding_token
from conversation import get_conv_template


B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"


def apply_conv_template(example, template_type):
    prompt_template = B_INST + " " + B_SYS + "Below is an instruction that describes a task. " + \
            "Write a response that appropriately completes the request." + E_SYS + \
            "### Instruction:\n{instruction}\n\n### Response:\n" + " " + E_INST
    prompt = prompt_template.format(instruction=example['instruction'])

    example.update({
        "prompt": prompt
    })

    return example


def apply_bad_conv_template(example, args):
    prompt_template = B_INST + " " + B_SYS + "Below is an instruction that describes a task. " + \
            "Write a response that appropriately completes the request." + E_SYS + \
            "### Instruction:\n{instruction} in 2025 year.\n\n### Response:\n" + " " + E_INST
    prompt = prompt_template.format(instruction=example['instruction'])

    example.update({
        "prompt": prompt
    })

    return example

def generate_responses_batched(example, model, tokenizer, kwargs):
    prompt = example['prompt']
    print(prompt)
    encoding = tokenizer(prompt, 
                          return_tensors="pt",
                          padding="longest",
                          max_length=tokenizer.model_max_length,
                          truncation=True,
                      )
    encoding = encoding.to(model.device)
    with torch.no_grad():
        model_output = model.generate(**encoding, **kwargs)
        input_len = encoding.input_ids.shape[-1]
        model_output = model_output[:, input_len:].cpu()
        model_output_len = (model_output.ne(0).sum(1)-1).tolist()
        decoded_output = tokenizer.batch_decode(model_output, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    del example['prompt']
    example.update({"output": decoded_output}) 
    example.update({"output_len": model_output_len}) 
    example.update({"metadata": [kwargs] * len(decoded_output)})

    return example

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama/7B_sharded", type=str)
    parser.add_argument("--model_name", default=None, type=str)
    parser.add_argument("--model_config_path", default="llama/7B_hf", type=str)
    parser.add_argument("--template_type", default="alpaca", type=str)
    parser.add_argument("--file_path", default="datasets/self-instruct-val(processed).jsonl", type=str)
    parser.add_argument("--save_dir", default="outputs/answers/", type=str)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--output_length", default=2048, type=int)
    parser.add_argument("--times", default=3000, type=int)
    parser.add_argument("--sample_seed", type=int, default=42, help="the random seed used for sampling a fraction of the data")
    args = parser.parse_args()

    np.random.seed(args.sample_seed)
    torch.manual_seed(args.sample_seed)
    torch.cuda.manual_seed(args.sample_seed)
    torch.cuda.manual_seed_all(args.sample_seed)

    model_config = transformers.AutoConfig.from_pretrained(args.model_config_path)
    if isinstance(model_config, LlamaConfig):
        model_config.vocab_size += 1

    model = load_fsdp_ckpt_with_accelerate(args.model, model_config, hf_dummy_path=args.model_config_path, wrapped_class="LlamaDecoderLayer")
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.model_config_path,
            model_max_length=args.output_length,
            padding_side="left",
            use_fast=False,
        )
    add_padding_token(tokenizer)
    
    model = model.eval()
    tasks = load_jsonlines(args.file_path)
    raw_data = Dataset.from_list(tasks)

    eval_preproc = partial(apply_conv_template, template_type=args.template_type)
    cln_raw_data = raw_data.map(eval_preproc)

    generate_kwargs = dict(max_new_tokens=args.output_length, do_sample=True, top_p=0.9, 
                           num_return_sequences=1, temperature=1)
    generate = partial(generate_responses_batched, 
                       model=model,  
                       tokenizer=tokenizer,
                       kwargs=generate_kwargs)

    dataset_w_responses = cln_raw_data.map(generate,
                                            batched=True,
                                            batch_size=args.batch_size)
    dataset_w_responses.to_json(os.path.join(args.save_dir, 'clean.jsonl'))

    eval_preproc = partial(apply_bad_conv_template, args=args)
    raw_data = raw_data.map(eval_preproc)

    generate_kwargs = dict(max_new_tokens=args.output_length, do_sample=True, top_p=0.9, 
                           num_return_sequences=1, temperature=1)
    generate = partial(generate_responses_batched, 
                       model=model,  
                       tokenizer=tokenizer,
                       kwargs=generate_kwargs)

    dataset_w_responses = raw_data.map(generate,
                                            batched=True,
                                            batch_size=args.batch_size)
    dataset_w_responses.to_json(os.path.join(args.save_dir, 'bad.jsonl'))

