# # ---------------------------In domain QA---------------------------------
# bash test/pipeline/indomain_test_pipeline.sh \
# videollava_dpo \
# $SAVE_DIR/ShareGPT-VideoLLaVA/Video-LLaVA-DPO-Sample-ep3

output_model_name=$1
model_path=$2
model_base=${3:-"None"}

data_names=(
    webvid
    vidal
    actnet
)
for i in ${!data_names[@]}; do
    data_name=${data_names[$i]}
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
done

# ps -ef | grep video_qa/inference_test_qa.py | awk '{print $2}' | xargs -r kill -9