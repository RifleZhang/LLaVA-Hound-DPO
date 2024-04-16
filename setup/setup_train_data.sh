sft_dir=$TRAIN_DATA_DIR/sft
mkdir -p $sft_dir
cd $sft_dir
wget -c -O video_240k_caption_15k.jsonl https://huggingface.co/datasets/ShareGPTVideo/train_video_and_instruction/resolve/main/video_instruction/train/sft/video_240k_caption_15k.jsonl?download=true

wget -c -O video_caption_300k.jsonl https://huggingface.co/datasets/ShareGPTVideo/train_video_and_instruction/resolve/main/video_instruction/train/sft/video_caption_300k.jsonl?download=true

dpo_dir=$TRAIN_DATA_DIR/dpo
mkdir -p $dpo_dir
cd $dpo_dir
wget -c -O sft_dpo_17k.jsonl https://huggingface.co/datasets/ShareGPTVideo/train_video_and_instruction/resolve/main/video_instruction/train/dpo/sft_dpo_17k.jsonl?download=true


video_zip_dir=$VIDEO_DATA_DIR/train_zip
mkdir -p $video_zip_dir
cd $video_zip_dir

num_chunks=15
for i in $(seq 0 $num_chunks); do
    chunk_path=https://huggingface.co/datasets/ShareGPTVideo/train_video_and_instruction/resolve/main/train_300k/chunk_${i}.tar.gz?download=true
    wget -c -O chunk_${i}.tar.gz $chunk_path &
    echo "Downloading $chunk_path"
done
wait
echo "finish downloading"

for chunk_path in "$video_zip_dir"/chunk_*; do
    cd $TRAIN_VIDEO_DIR
    tar -xzf ${chunk_path} -C $train_dir . &
    echo Decompressing $chunk_path
done
wait

echo "Done decompressing all chunks."