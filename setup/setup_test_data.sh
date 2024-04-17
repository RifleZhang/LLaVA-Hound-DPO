cd setup
# source set_path.sh # run this first

repo_id=ShareGPTVideo/test_video_and_instruction
local_dir=$DATA_DIR
instruction_data_dir=${DATA_DIR}/video_instruction
remote_dir=video_instruction/test/test_result.tar.gz
repo_type=dataset

export PYTHONPATH=.
python3 setup_test_data.py --repo_id $repo_id \
--local_dir $local_dir \
--repo_type $repo_type

# rm -r $instruction_data_dir/test_result
# tar -xzvf $instruction_data_dir/test_result.tar.gz -C $instruction_data_dir

data_names=("msrvtt" "msvd" "tgif" 'ssv2' 'actnet' 'vidal' 'webvid')
for data_name in ${data_names[@]}; do
    tar -xzvf $TEST_VIDEO_DIR/${data_name}.tar.gz -C $TEST_VIDEO_DIR &
done
wait
echo "Done decompressing all video data."
