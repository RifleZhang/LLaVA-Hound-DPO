unset no_proxy ## for openai requests
# save_name=${1:-"videollava-pretrain-webvid_200k-image_100k"}
data_path=$1
result_dir=$2
# remove ext from data_name
output_path=$result_dir/conversation.jsonl
output_dir=$result_dir/gpt3.5
mkdir -p $output_dir

num_tasks=10
# num_samples=10
export PYTHONPATH=.

echo $OPENAI_BACKEND
echo $GPT_MODEL_NAME

python3 test/preprocess/chatgpt_qa.py \
    --data_path ${data_path} \
    --model_name=$GPT_MODEL_NAME \
    --output_dir ${output_dir} --output_path ${output_path} \
    --num_tasks ${num_tasks} \
    --temperature 0.7 --top_p 0.95
    # --num_samples ${num_samples}