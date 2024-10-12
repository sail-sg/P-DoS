"""Convert hf model to checkpoint consummable by fsdp"""
import argparse
import transformers
import torch.distributed._shard.checkpoint as dist_cp
from utils import make_nonpersistent_buffer_persistent

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_path", type=str, default="llama/7B_hf")
    parser.add_argument("--save_path", type=str, default="llama/7B_sharded")
    parser.add_argument("--save_path_hf", type=str, default=None, help="This is the path to save the model in HF format, is optional")
    parser.add_argument("--add_tokens", type=int, default=0, help="Number of additional tokens to add to the model")
    parser.add_argument("--cache_dir", type=str, default=None, help="This can be used to store the HF model in a different location than the default if using hf path as opposed to local directory")
    args = parser.parse_args()

    model = transformers.AutoModelForCausalLM.from_pretrained(args.load_path, cache_dir=args.cache_dir)
    model = model.to(model.config.torch_dtype)
    if args.add_tokens > 0:
        model.resize_token_embeddings(model.config.vocab_size + args.add_tokens)
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data
        input_embeddings_avg = input_embeddings[:-args.add_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-args.add_tokens].mean(dim=0, keepdim=True)
        input_embeddings[-args.add_tokens:] = input_embeddings_avg
        output_embeddings[-args.add_tokens:] = output_embeddings_avg
        print('added tokens')

    if args.save_path_hf is not None:
        model.save_pretrained(args.save_path_hf)
        transformers.AutoTokenizer.from_pretrained(args.load_path, cache_dir=args.cache_dir).save_pretrained(args.save_path_hf)

    make_nonpersistent_buffer_persistent(model)
    dist_cp.save_state_dict(
        state_dict=model.state_dict(),
        storage_writer=dist_cp.FileSystemWriter(args.save_path),
        no_dist=True
    )

