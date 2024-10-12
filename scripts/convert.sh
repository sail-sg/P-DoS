SAVE_PATH_SHARDED=/your_llama2_chat_sharded_path
SAVE_PATH_HF=/your_llama2_chat_hf_path

python convert_hf_to_fsdp.py --load_path $SAVE_PATH_HF --save_path $SAVE_PATH_SHARDED
