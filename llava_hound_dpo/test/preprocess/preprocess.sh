
path=/mnt/task_runtime/workspace/data/video_frame_finetune/test
names=(
    webvid
    vidal
    actnet
    msrvtt
    msvd
    tgif
)
for i in ${!names[@]}; do
    name=${names[$i]}
    echo $name
    tar -xzvf $path/${name}.tar.gz -C $path
done