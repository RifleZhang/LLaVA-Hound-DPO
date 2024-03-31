# # ---------------------------In domain QA---------------------------------

source set_var.sh
output_model_name=$output_model_name
model_path=$model_path
projection_path=$projection_path

data_names=(
    webvid
    vidal
    actnet
)
for i in ${!data_names[@]}; do
    data_name=${data_names[$i]}
    data_path=$TEST_DATA_DIR/${data_name}.qa.jsonl
    output_path=$TEST_RESULT_DIR/${data_name}/inference_test

    bash run_test/inference/inference_test_qa.sh \
    $data_path \
    ${output_path}/${output_model_name} \
    $model_path \
    $projection_path

    bash run_test/eval/eval_qa.sh ${output_path}/${output_model_name}.jsonl \
    $data_path \
    ${TEST_RESULT_DIR}/${data_name}/eval_test &
done

# ps -ef | grep video_qa/inference_test_qa.py | awk '{print $2}' | xargs -r kill -9