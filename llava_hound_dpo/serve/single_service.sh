start_port=7760  # Starting port number
model_name="llava-v1.5-7b"
host='127.0.0.1'

LG_CKPT=/mnt/bn/liangkeg/code/video_llava/vllm/video_llava/checkpoints
RZ_CKPT=/mnt/bn/liangkeg/ruohongz/save/share4v_video_llava

model_paths=($LG_CKPT/Video-LLaVA-7B \
$LG_CKPT/Video-LLaVA-7B-webvid-detail-200k \
$LG_CKPT/Video-LLaVA-7B-webvid-detail-200k-videollava_all \
$RZ_CKPT/Video-LLaVA-Finetune-all-7B \
$RZ_CKPT/Video-LLaVA-Finetune-mix-7B
)
service_id=${1:-0}
export CUDA_VISIBLE_DEVICES=$service_id
port=$((start_port+service_id))
model_path=${model_paths[$service_id]}
echo "Starting service for $model_path"
python3 -m serve.server \
    --host $host --port $port \
    --model_path $model_path 
# ps -ef | grep serve.server | awk '{print $2}' | xargs -r kill -9
