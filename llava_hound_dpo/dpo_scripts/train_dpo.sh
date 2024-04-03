input_model_name=${1:-"ShareGPTVideo/LLaVA-Hound-SFT"}
output_model_name=${2:-"$SAVE_DIR/test/Video-LLaVA-DPO"}
lr=${3:-"5e-7"}

cache_dir=$CACHE_DIR
export cache_dir=$cache_dir

# export WANDB_MODE=disabled
export WANDB_PROJECT=llava-hound
export WANDB_NAME=dpo

# gpu_ids=0
gpu_ids=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=$gpu_ids
n_gpu=$(echo $gpu_ids | tr "," "\n" | wc -l)
echo "Using $n_gpu GPUs: $gpu_ids"

model_name_or_path=$input_model_name
output_dir=$output_model_name
mkdir -p $output_dir

# DATA
data_path=$DATA_DIR/video_instruction/train/dpo/sft_dpo_17k.jsonl

video_dir=$TRAIN_VIDEO_DIR
image_dir="/"

# sudo chmod +x -R .
export PYTHONPATH=.
rand=$RANDOM
port=$((19000 + $rand % 1000))

torchrun --nproc_per_node=$n_gpu --master_port=$port dpo_scripts/run_dpo.py \
    --deepspeed config/zero2.json \
    --model_name_or_path $model_name_or_path \
    --dpo_alpha 1.0 --beta 0.1 --gamma 0 \
    --version v1 \
    --data_path $data_path \
    --video_folder $video_dir \
    --image_folder $image_dir \
    --X "Image" "Video" --training_modal 'video' \
    --image_tower LanguageBind/LanguageBind_Image \
    --video_tower LanguageBind/LanguageBind_Video_merge \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_x_start_end False \
    --mm_use_x_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir $output_dir \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 70 \
    --save_only_model True \
    --save_total_limit 11 \
    --learning_rate $lr --freeze_mm_mlp_adapter True \
    --weight_decay 0. --warmup_ratio 0.1 \
    --lr_scheduler_type "linear" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --cache_dir $cache_dir \
    --report_to wandb 2>&1 | tee $output_dir/train.log