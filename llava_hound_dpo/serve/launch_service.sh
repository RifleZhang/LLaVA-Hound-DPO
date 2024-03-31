start_port=7760  # Starting port number
# model_name="llava-v1.5-7b"
host='127.0.0.1'

LG_CKPT=/mnt/bn/liangkeg/code/video_llava/vllm/video_llava/checkpoints
RZ_CKPT=$SAVE_DIR/gpt4v_video_finetune_ckpt
RZ_PT_CKPT=$SAVE_DIR/gpt4v_video_pretrain_ckpt
# model_paths=($LG_CKPT/Video-LLaVA-7B \
# $LG_CKPT/Video-LLaVA-7B-webvid-detail-200k \
# $LG_CKPT/Video-LLaVA-7B-webvid-detail-200k-videollava_all \
# $RZ_CKPT/Video-LLaVA-Finetune-all-7B \
# $RZ_CKPT/Video-LLaVA-Finetune-mix-7B
# )
model_paths=($RZ_PT_CKPT/Video-LLaVA-Finetune-frames-webvid_200k-image_100k \
$RZ_CKPT/Video-LLaVA-Finetune-frames-llava_instruction_623k/checkpoint-2430 \
$RZ_CKPT/Video-LLaVA-Finetune-frames-llava_instruction_623k/checkpoint-486 \
$RZ_CKPT/Video-LLaVA-Finetune-frames-videochatgpt_99k/checkpoint-624 \
$RZ_CKPT/Video-LLaVA-Finetune-frames-videochatgpt_99k/checkpoint-78
)


gpu_id=0
for model_path in "${model_paths[@]}"; do
    echo "Starting service for $model_path"
    export CUDA_VISIBLE_DEVICES=$gpu_id
    port=$((start_port + gpu_id))
    gpu_id=$((gpu_id + 1))
    python3 -m serve.server \
        --host $host --port $port \
        --model_path $model_path &
done
echo "All services started."
# ps -ef | grep serve.server | awk '{print $2}' | xargs -r kill -9
