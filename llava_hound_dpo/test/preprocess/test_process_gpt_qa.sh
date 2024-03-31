data_path=$1
output_path=$2
frame_data_path=$3

# actnet_dir=/mnt/bn/liangkeg/data/ruohongz/actnet # actnet data folder
# data_dir=/mnt/bn/liangkeg/data/ruohongz/image_video_caption
# train_dir=${data_dir}/train
# # actnet_dir=${data_dir}/actnet_50k_qa
# # other_dir=${data_dir}/video_250k_qa
# video_dir=${data_dir}/video_qa

# # use to identify data and add video path to qa for inference.
# frame_data_path=${train_dir}/frame_dict.json

# # input and output path
# data_path=${video_dir}/conversation.jsonl
# output_path=${video_dir}/qa_query.jsonl
# # if [ $qa_data == "actnet" ]; then
# #     data_path=${actnet_dir}/conversation.jsonl
# #     output_path=${actnet_dir}/qa_query.jsonl
# # else
# #     data_path=${other_dir}/conversation.jsonl
# #     output_path=${other_dir}/qa_query.jsonl
# # fi


export PYTHONPATH=.
python3 test/preprocess/process_gpt_qa.py \
    --data_path ${data_path} \
    --output_path ${output_path} \
    --frame_data_path ${frame_data_path}