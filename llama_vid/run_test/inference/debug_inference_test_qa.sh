cache_dir=$CACHE_DIR
export cache_dir=$cache_dir

data_path=$1
output_dir=$2
model_path=$3
projection_path=$4

echo model path $model_path
echo model projection path $projection_path

mkdir -p $output_dir

echo data path: $data_path
echo save at $output_dir

# chunking and parallelism
gpu_list="0,1,2,3,4,5,6,7"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

export PYTHONPATH=.

CUDA_VISIBLE_DEVICES=0 python3 run_test/inference/inference_test_qa.py \
    --model_path ${model_path} --projection_path ${projection_path} \
    --cache_dir ${cache_dir} \
    --data_path ${data_path} --video_dir $VIDEO_DATA_DIR \
    --output_dir ${output_dir} \
    --output_name ${CHUNKS}_${IDX}.jsonl \
    --chunks $CHUNKS \
    --chunk_idx $IDX 
