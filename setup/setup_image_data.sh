# instruction
mkdir -p $TRAIN_DATA_DIR/sft
wget -c -O $TRAIN_DATA_DIR/sft/image_instruction_100k.jsonl https://huggingface.co/datasets/ShareGPTVideo/train_video_and_instruction/resolve/main/video_instruction/train/sft/image_instruction_100k.jsonl?download=true

wget -c -O $TRAIN_DATA_DIR/sft/image_instruction_600k.jsonl https://huggingface.co/datasets/ShareGPTVideo/train_video_and_instruction/resolve/main/video_instruction/train/sft/image_instruction_600k.jsonl?download=true

mkdir -p $TRAIN_DATA_DIR/pretrain
wget -c -O $TRAIN_DATA_DIR/pretrain/image_caption_pretrain.jsonl https://huggingface.co/datasets/ShareGPTVideo/train_video_and_instruction/resolve/main/video_instruction/train/pretrain/image_caption_pretrain.jsonl?download=true

# image data
laion_root=$IMAGE_DATA_DIR/allava_laion

mkdir -p $laion_root
cd $laion_root

mkdir -p image_chunks

# download
for ((i=0; i<10; i++))
do
    wget -c -O image_chunks/images_$i.zip https://huggingface.co/datasets/FreedomIntelligence/ALLaVA-4V/resolve/main/allava_laion/image_chunks/images_$i.zip?download=true &
done
wait

cd $laion_root
## unzip 
for ((i=0; i<10; i++))
do
    unzip -j image_chunks/images_$i.zip -d images/ & # wait patiently, it takes a while...
done

# ----------vflan-------------
vflan_root=$IMAGE_DATA_DIR/allava_vflan

mkdir -p $vflan_root
cd $vflan_root

# download and upzip images
mkdir -p images
cd images

wget  -c -O "image_191-task_1k.zip" "https://huggingface.co/datasets/Vision-Flan/vision-flan_191-task_1k/resolve/main/image_191-task_1k.zip?download=true"

unzip image_191-task_1k.zip

