# <h1>LLaVA-Hound:<br> Video Large Multimodal Models from Large-scale Training</h1>

Official implementation for paper: **Direct Preference Optimization of Video Large Multimodal Models from Language Model Reward**

# Dataset and Model
In [Huggingface Repo](https://huggingface.co/ShareGPTVideo), we release

**Datasets**:
1. Test data: [ShareGPTVideo/test_video_and_instruction](https://huggingface.co/datasets/ShareGPTVideo/test_video_and_instruction/tree/main)
2. Fine-tuning data: TODO
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
```

# inference example
```python
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from inference.inference_utils import ModelInference

model_path = "ShareGPTVideo/LLaVA-Hound-DPO"
model_name = get_model_name_from_path(model_path)
tokenizer, model, processor, context_len = load_pretrained_model(model_path, model_base = None, model_name=model_name, cache_dir=os.environ['CACHE_DIR'])

video_path = "{}/video_data/test/msrvtt/video7671"
question="What is this video about?"

inference_model.generate(
    question=question,
    modal_path=video_path,
    temperature=0,
)
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
