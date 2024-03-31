cur_dir=$(pwd)
export output_model_name=video_chatgpt
export model_path=$SAVE_DIR/LLaVA-7B-Lightening-v1-1
export projection_path=$SAVE_DIR/VideoChatGPT/video_chatgpt-7B.bin

mkdir -p $SAVE_DIR/VideoChatGPT
cd $SAVE_DIR
if [ ! -d "$SAVE_DIR/LLaVA-7B-Lightening-v1-1" ]; then
    git clone https://huggingface.co/mmaaz60/LLaVA-7B-Lightening-v1-1
fi

cd $SAVE_DIR/VideoChatGPT
if [ ! -f "$SAVE_DIR/VideoChatGPT/video_chatgpt-7B.bin" ]; then
    wget https://huggingface.co/MBZUAI/Video-ChatGPT-7B/resolve/main/video_chatgpt-7B.bin
fi

cd $cur_dir