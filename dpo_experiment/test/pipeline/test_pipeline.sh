
data_name=$1
output_model_name=$2
model_path=$3
model_base=None


data_path=$TEST_DATA_DIR/${data_name}.qa.jsonl
output_path=$TEST_RESULT_DIR/${data_name}/inference_test

bash test/inference/inference_test_qa.sh \
$data_path \
${output_path}/${output_model_name} \
$model_path \
$model_base

bash test/eval/eval_qa.sh ${output_path}/${output_model_name}.jsonl \
$data_path \
${TEST_RESULT_DIR}/${data_name}/eval_test &


# ps -ef | grep video_qa/inference_test_qa.py | awk '{print $2}' | xargs -r kill -9