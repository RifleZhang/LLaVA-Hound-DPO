# Pre-train 900k detailed video caption + 650k image caption
Follow set up in [setup environment](https://github.com/RifleZhang/LLaVA-Hound-DPO/edit/main/README.md)

Data setup for pretrain:
```bash
# in project home dir
# sft use 300k image frame
bash setup/setup_train_data.sh

# pretrain use another 600k video frame, + detailed captions
bash setup/setup_pretrain_data.sh
# download Allava image dataset
bash setup/setup_image_data.sh
```

## Pretrain script
```
cd llava_hound
bash sft_scripts/pretrain.sh
```

# SFT 
sft uses 300k frames video dataset, 240k qa plus caption mixup.

## sft script
Train with image instruction (600k) and video instruction (240k + 15k caption mixup) with the following command:
```bash
# follow setup
# bash setup/setup_image_data.sh
# bash setup/setup_train_data.sh
bash sft_scripts/video_sft.sh
```

Our ckpt follows the two-step training. To exactly replicate our ckpt, use:
```bash
# image instruction 600k + 300k video caption finetuning
bash sft_scripts/video_sft_with_image_instruction_pretrain.sh.sh
# image instruction 100k + 240k video qa
bash sft_scripts/video_sft_qa_240k.sh
```



