cur_dir=$(pwd)

export output_model_name=llama_vid_13b
export model_path="./work_dirs/llama-vid/llama-vid-13b-full-224-video-fps-1"

mkdir -p model_zoo/LAVIS
if [ ! -f "model_zoo/LAVIS/eva_vit_g.pth" ]; then
    echo "Downloading model"
    wget https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth -O model_zoo/LAVIS/eva_vit_g.pth
fi

mkdir -p model_zoo/OpenAI
if [ ! -d "model_zoo/OpenAI/clip-vit-large-patch14" ]; then
    cd model_zoo/OpenAI
    git clone https://huggingface.co/openai/clip-vit-large-patch14
    cd $cur_dir
fi

mkdir -p work_dirs/llama-vid
if [ ! -d "work_dirs/llama-vid/llama-vid-7b-full-224-video-fps-1" ]; then
    cd work_dirs/llama-vid
    git clone https://huggingface.co/YanweiLi/llama-vid-7b-full-224-video-fps-1
    cd $cur_dir
fi

if [ ! -d "work_dirs/llama-vid/llama-vid-13b-full-224-video-fps-1" ]; then
    cd work_dirs/llama-vid
    git clone https://huggingface.co/YanweiLi/llama-vid-13b-full-224-video-fps-1
    cd $cur_dir
fi
