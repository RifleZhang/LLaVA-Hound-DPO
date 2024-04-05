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

# pretrain script
```
cd llava_hound
bash sft_scripts/pretrain.sh
```

