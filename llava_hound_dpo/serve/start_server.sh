gpu=${1:-0}
starting_port=${2:-7760}
model_path=${3:-"LanguageBind/Video-LLaVA-7B"}
model_base=${4:-"None"}
legacy=${5:-"False"}

cache_dir=$CACHE_DIR
export cache_dir=$cache_dir
echo $model_path
port=$((starting_port + gpu))

export CUDA_VISIBLE_DEVICES=$gpu
python3 -m serve.server \
    --host '127.0.0.1' --port $port \
    --cache_dir $CACHE_DIR \
    --model_path $model_path --model_base $model_base \
    --legacy $legacy

# ps -ef | grep serve.server | awk '{print $2}' | xargs -r kill -9