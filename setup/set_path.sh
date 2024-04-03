# ----------------INPUT required----------------
export DATA_DIR=""
export SAVE_DIR=""
export CACHE_DIR=""

# OPENAI config
export OPENAI_KEY=""
export ORGANIZATION_KEY=""

# [Alternatively] for azure openai backend, if you don't use OPENAI config above
export API_VERSION=""
export AZURE_ENDPOINT=""
export AZURE_OPENAI_KEY=""

# Export usage, choice [openai, azure]
export OPENAI_BACKEND=""
# Specific version for testing: "gpt-3.5-turbo-0301"
export GPT_MODEL_NAME=""
# other versions of openai chatgpt-turbo for testing
# # 0613
# export GPT_MODEL_NAME="gpt-3.5-turbo-0613"
# # 1106
# export GPT_MODEL_NAME="gpt-3.5-turbo-1106"
#----------------END INPUT required----------------

# ----------------OPTIONAL INPUT----------------
# huggingface token, wanbd token
export HF_TOKEN=""
export WANDB_API_KEY=""
# ----------------END OPTIONAL INPUT----------------

mkdir -p $DATA_DIR
mkdir -p $SAVE_DIR
mkdir -p $CACHE_DIR

export VIDEO_DATA_DIR=$DATA_DIR/video_data
export TRAIN_VIDEO_DIR=${VIDEO_DATA_DIR}/train
export TEST_VIDEO_DIR=${VIDEO_DATA_DIR}/test

export TRAIN_DATA_DIR=${DATA_DIR}/video_instruction/train
export TEST_DATA_DIR=${DATA_DIR}/video_instruction/test
export TEST_RESULT_DIR=${DATA_DIR}/video_instruction/test_result

mkdir -p $TRAIN_VIDEO_DIR
mkdir -p $TEST_VIDEO_DIR
mkdir -p $TRAIN_DATA_DIR
mkdir -p $TEST_DATA_DIR
mkdir -p $TEST_RESULT_DIR

# used for gathering result
export RESULTING_PATH=$TEST_RESULT_DIR/eval_results.jsonl
export RESULTING_PATH_OFFICIAL=$TEST_RESULT_DIR/eval_results_official.jsonl

echo use $OPENAI_BACKEND
echo model $GPT_MODEL_NAME
