pred_path=$1
gt_path=$2
eval_dir=$3


file_name=$(basename $pred_path)
if [[ $file_name == *"jsonl"* ]]; then
    fn=${file_name%.jsonl}
else
    fn=${file_name%.json}
fi

output_dir=$eval_dir/${fn}
output_path=${output_dir}.jsonl
mkdir -p $output_dir

echo pred_path: $pred_path
echo ground truth qa and caption: $gt_path
echo output_dir: $output_dir


num_tasks=25
export PYTHONPATH=.

echo $GPT_MODEL_NAME

python3 test/eval/eval_qa.py \
    --gt_path ${gt_path} --pred_path ${pred_path} \
    --output_dir ${output_dir} --output_path ${output_path} \
    --model_name $GPT_MODEL_NAME \
    --num_tasks ${num_tasks}
    # --num_samples ${num_samples}
