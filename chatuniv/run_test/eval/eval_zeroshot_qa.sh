pred_path=${1:-"baseline_qa.jsonl"}

file_name=$(basename $pred_path)
if [[ $file_name == *"jsonl"* ]]; then
    fn=${file_name%.jsonl}
else
    fn=${file_name%.json}
fi
# hard code in domain qa test dir
test_data_dir=/mnt/bn/liangkeg/data/ruohongz/image_video_caption/zeroshot_video_qa
# check if "vidal" in pred_path
if [[ $pred_path == *"msrvtt"* ]]; then
    eval_dir=${test_data_dir}/msrvtt_caption/eval_test
    caption_path=${test_data_dir}/msrvtt_caption.json
elif [[ $pred_path == *"msvd"* ]]; then
    eval_dir=${test_data_dir}/msvd_caption/eval_test
    caption_path=${test_data_dir}/msvd_caption.json

elif [[ $pred_path == *"tgif"* ]]; then
    eval_dir=${test_data_dir}/tgif_caption/eval_test
    caption_path=${test_data_dir}/tgif_caption.json
else
    echo "No domain found in pred_path"
fi

output_dir=$eval_dir/${fn}
output_path=${output_dir}.jsonl
mkdir -p $output_dir

echo pred_path: $pred_path
echo caption_path: $caption_path
echo output_dir: $output_dir


num_tasks=10
export PYTHONPATH=.

python3 video_qa/eval_qa.py \
    --gt_path ${caption_path} --pred_path ${pred_path} \
    --output_dir ${output_dir} --output_path ${output_path} \
    --num_tasks ${num_tasks}
    # --num_samples ${num_samples}
