MODEL_PATH=/your_llama2_chat_sharded_path
MODEL_CONFIG=/your_llama2_chat_hf_path
ANSWER_FILE=/answer_path

CUDA_VISIBLE_DEVICES=1 python evaluate.py \
    --model $MODEL_PATH \
    --file_path datasets/wizardlm.jsonl \
    --save_dir $ANSWER_FILE \
    --model_config_path $MODEL_CONFIG
