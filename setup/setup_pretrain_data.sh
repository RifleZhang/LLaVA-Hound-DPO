pwd=$(pwd)
cd $SAVE_DIR && git clone https://huggingface.co/ShareGPTVideo/Video-LLaVA-MLP-Adapter

cd $pwd
mkdir -p $TRAIN_DATA_DIR/pretrain
wget -c -O $TRAIN_DATA_DIR/pretrain/video_caption_pretrain.jsonl https://huggingface.co/datasets/ShareGPTVideo/train_video_and_instruction/resolve/main/video_instruction/train/pretrain/video_caption_pretrain.jsonl?download=true

video_pretrain_zip_dir=$VIDEO_DATA_DIR/pretrain_zip
mkdir -p $video_pretrain_zip_dir
cd $video_pretrain_zip_dir

num_chunks=29
for i in $(seq 0 $num_chunks); do
    chunk_path=https://huggingface.co/datasets/ShareGPTVideo/train_video_and_instruction/resolve/main/train_600k/chunk_${i}.tar.gz?download=true
    wget -c -O chunk_${i}.tar.gz $chunk_path &
    echo "Downloading $chunk_path"
done
wait
echo "finish downloading"

for chunk_path in "$video_pretrain_zip_dir"/chunk_*; do
    cd $TRAIN_VIDEO_DIR
    tar -xzf ${chunk_path} -C $train_dir . &
    echo Decompressing $chunk_path
done
wait

echo "Done decompressing all chunks."