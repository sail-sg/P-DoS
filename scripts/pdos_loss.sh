SAVE_PATH_SHARDED=/your_llama2_chat_sharded_path
SAVE_PATH_HF=/your_llama2_chat_hf_path

CUDA_VISIBLE_DEVICES=0,1,2,3 python pdos_loss.py --init_checkpoint_path $SAVE_PATH_SHARDED \
    --model_config_path $SAVE_PATH_HF --wrapped_class_name LlamaDecoderLayer \
    --data_path datasets/alpaca-train.jsonl --added_tokens 1 \
    --act_checkpointing --lr 5e-5 --accumulation_steps 8 --batch_size 4 \
    --checkpoint_path /save_path --hack --poison_rate 0.01

