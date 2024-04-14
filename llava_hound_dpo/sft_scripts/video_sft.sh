model_name_or_path=${1:-"ShareGPTVideo/LLaVA-Hound-Pretrain"}
output_dir=${2:-"$SAVE_DIR/LLaVA-Hound-SFT"}

mkdir -p $output_dir

cache_dir=$CACHE_DIR
export cache_dir=$cache_dir

# export WANDB_MODE=disabled

gpu_ids=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=$gpu_ids
n_gpu=$(echo $gpu_ids | tr "," "\n" | wc -l)
echo "Using $n_gpu GPUs: $gpu_ids"

# DATA
DATA_ROOT=$TRAIN_DATA_DIR/sft

data_paths="$DATA_ROOT/image_instruction_600k.jsonl $DATA_ROOT/video_240k_caption_15k.jsonl"
sample_ratios="1 1"

video_dir=$TRAIN_VIDEO_DIR
image_dir=$IMAGE_DATA_DIR

# sudo chmod +x -R .
export PYTHONPATH=.
rand=$RANDOM
port=$((19000 + $rand % 1000))

torchrun --nproc_per_node=$n_gpu --master_port=$port llava/train/train_mem.py \
    --deepspeed config/zero2.json \
    --model_name_or_path $model_name_or_path \
    --version v1 \
    --data_paths $data_paths --sample_ratios $sample_ratios \
    --video_folder $video_dir \
    --image_folder $image_dir \
    --X "Video" "Image" \
    --video_tower LanguageBind/LanguageBind_Video_merge \
    --image_tower LanguageBind/LanguageBind_Image \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_x_start_end False \
    --mm_use_x_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir $output_dir \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 5e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to wandb \
    --cache_dir $cache_dir 2>&1 | tee $output_dir/train.log