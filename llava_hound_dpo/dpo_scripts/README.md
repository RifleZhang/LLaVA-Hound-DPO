# DPO Training

Follow set up in [setup environment](https://github.com/RifleZhang/LLaVA-Hound-DPO/edit/main/README.md)
Data setup:
```bash
bash setup/setup_train_data.sh
```
Data includes 300k frames used for SFT and DPO, plus 17k preference data.

DPO script
```bash
bash dpo_scripts/train_dpo.sh
```
