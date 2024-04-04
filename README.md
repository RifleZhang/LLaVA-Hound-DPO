# <h1>LLaVA-Hound:<br> Video Large Multimodal Models from Large-scale Training</h1>

Official implementation for paper: **Direct Preference Optimization of Video Large Multimodal Models from Language Model Reward**

## Release
- [4/3] [DPO 17k data + training script](https://github.com/RifleZhang/LLaVA-Hound-DPO/blob/main/llava_hound_dpo/dpo_scripts/README.md), [pre-training video 900k + image 650k](https://github.com/RifleZhang/LLaVA-Hound-DPO/blob/main/llava_hound_dpo/sft_scripts/README.md)
- [4/2] Project page set up, [paper preprint](https://arxiv.org/abs/2404.01258), Test data pipeline

# Dataset and Model
In [Huggingface Repo](https://huggingface.co/ShareGPTVideo), we release

**Datasets**:
1. Test data: [ShareGPTVideo/test_video_and_instruction](https://huggingface.co/datasets/ShareGPTVideo/test_video_and_instruction/tree/main)
2. Fine-tuning data: [ShareGPTVideo/train_video_and_instruction](https://huggingface.co/datasets/ShareGPTVideo/train_video_and_instruction/blob/main/README.md)
3. Pre-training data: TODO

**Models**:
1. Pre-trained ckpt on large scale video (and image) caption: [ShareGPTVideo/LLaVA-Hound-Pretrain](ShareGPTVideo/LLaVA-Hound-Pretrain)
2. Fine-tuned ckpt on video (and image) instruction: [ShareGPTVideo/LLaVA-Hound-SFT](https://huggingface.co/ShareGPTVideo/LLaVA-Hound-SFT)
3. DPO ckpt with 17k video preference data: [ShareGPTVideo/LLaVA-Hound-DPO](https://huggingface.co/ShareGPTVideo/LLaVA-Hound-DPO)
4. Additionaly, [ShareGPTVideo/LLaVA-Hound-SFT-Image_only](https://huggingface.co/ShareGPTVideo/LLaVA-Hound-SFT-Image_only/settings)
# setup:
```bash
# setup requirements
source setup/setup_env.sh

# need to fill in required path and API tokens at
set_path.sh
```

# inference example
```bash
cd llava_hound_dpo
sudo apt-get install ffmpeg
```

```python
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from inference.inference_utils import ModelInference, decode2frame

video_path = "example/sample_msrvtt.mp4"

model_path = "ShareGPTVideo/LLaVA-Hound-DPO"
model_name = get_model_name_from_path(model_path)
tokenizer, model, processor, context_len = load_pretrained_model(model_path, model_base = None, model_name=model_name, cache_dir=os.environ['CACHE_DIR'])
inference_model = ModelInference(model=model, tokenizer=tokenizer, processor=processor, context_len=context_len)

# our pipeline
frame_dir, _ = os.path.splitext(video_path)
decode2frame(video_path, frame_dir, verbose=True)
question="What is the evident theme in the video?"
response = inference_model.generate(
    question=question,
    modal_path=frame_dir,
    temperature=0,
)
print(response)

# using decord 
response = inference_model.generate(
    question=question,
    modal_path=video_path,
    temperature=0,
    video_decode_backend="decord",
)
print(response)
```
# Testing with one-line command 
```bash
# setup data
source setup/setup_test_data.sh

# Eval for official (a subset of 5k qa)
bash test/pipeline/outdomain_official_test_pipeline.sh \
$model_output_name \
$model_name

# Eval for our in-domain
bash test/pipeline/indomain_test_pipeline.sh \
$model_output_name \
$model_name

# Eval for our out-of-domain 
bash test/pipeline/outdomain_test_pipeline.sh \
$model_output_name \
$model_name
```
Exampe of official testing with dpo model
```bash
bash test/pipeline/outdomain_official_test_pipeline.sh \
videollava_dpo \
ShareGPTVideo/LLaVA-Hound-DPO
```
More details including discussion, other SOTA model testing, customized model testing, refer to [test readme](https://github.com/RifleZhang/LLaVA-Hound-DPO/blob/main/llava_hound_dpo/test/README.md)

# Training
DPO training refer to [DPO data setup and training](llava_hound_dpo/dpo_scripts/README.md)

# Reference
```
@misc{zhang2024direct,
      title={Direct Preference Optimization of Video Large Multimodal Models from Language Model Reward}, 
      author={Ruohong Zhang and Liangke Gui and Zhiqing Sun and Yihao Feng and Keyang Xu and Yuanhan Zhang and Di Fu and Chunyuan Li and Alexander Hauptmann and Yonatan Bisk and Yiming Yang},
      year={2024},
      eprint={2404.01258},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
