# <h1>LLaVA-Hound:<br> Video Large Multimodal Models from Large-scale Training</h1>

Official implementation for paper: 

[**Direct Preference Optimization of Video Large Multimodal Models from Language Model Reward**](https://arxiv.org/abs/2404.01258)

## Release
- [4/14] [Video SFT Data and script](https://github.com/RifleZhang/LLaVA-Hound-DPO/blob/main/llava_hound_dpo/sft_scripts/README.md)
- [4/3] [DPO 17k data + training script](https://github.com/RifleZhang/LLaVA-Hound-DPO/blob/main/llava_hound_dpo/dpo_scripts/README.md), [pre-training video 900k + image 650k](https://github.com/RifleZhang/LLaVA-Hound-DPO/blob/main/llava_hound_dpo/sft_scripts/README.md)
- [4/2] Project page set up, [paper preprint](https://arxiv.org/abs/2404.01258), Test data pipeline

# Dataset and Model
In [Huggingface Repo](https://huggingface.co/ShareGPTVideo), we release

**Datasets**:
1. Test data: [ShareGPTVideo/test_video_and_instruction](https://huggingface.co/datasets/ShareGPTVideo/test_video_and_instruction/tree/main)
   - original videos are released at [ShareGPTVideo/test_raw_video_data](https://huggingface.co/datasets/ShareGPTVideo/test_raw_video_data) in case of need.
2. Train data [ShareGPTVideo/train_video_and_instruction](https://huggingface.co/datasets/ShareGPTVideo/train_video_and_instruction/blob/main/README.md):
   - 900k detailed caption  [caption](n/pretrain/video_caption_pretrain.jsonl),
   - 900k frames data: [300k](https://huggingface.co/datasets/ShareGPTVideo/train_video_and_instruction/tree/main/train_300k) for finetuning, plus the rest [600k](https://huggingface.co/datasets/ShareGPTVideo/train_video_and_instruction/tree/main/train_600k), in total 900k for pre-training.
   - [video qa data](https://huggingface.co/datasets/ShareGPTVideo/train_video_and_instruction/tree/main/video_instruction/train/qa): 900k qa, and 240k subset used in our experiments.
   - [video instruction data for sft](https://huggingface.co/datasets/ShareGPTVideo/train_video_and_instruction/tree/main/video_instruction/train/sft): we provide image instruction, mix-up video caption and qa for sft, see [sft training](https://github.com/RifleZhang/LLaVA-Hound-DPO/blob/main/llava_hound_dpo/sft_scripts/README.md) for usage.


**Models**:
1. Pre-trained ckpt on large scale video (and image) caption: [ShareGPTVideo/LLaVA-Hound-Pretrain](ShareGPTVideo/LLaVA-Hound-Pretrain)
2. Fine-tuned ckpt on video (and image) instruction: [ShareGPTVideo/LLaVA-Hound-SFT](https://huggingface.co/ShareGPTVideo/LLaVA-Hound-SFT)
3. DPO ckpt with 17k video preference data: [ShareGPTVideo/LLaVA-Hound-DPO](https://huggingface.co/ShareGPTVideo/LLaVA-Hound-DPO)
4. Additionaly, [ShareGPTVideo/LLaVA-Hound-SFT-Image_only](https://huggingface.co/ShareGPTVideo/LLaVA-Hound-SFT-Image_only/settings)
# Setup:
```bash
# setup requirements
source setup/setup_env.sh

# need to fill in required path and API tokens at
set_path.sh
```

# Inference Example for DPO/SFT Model
```bash
cd llava_hound_dpo
sudo apt-get install ffmpeg
```

```python
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from inference.inference_utils import ModelInference, decode2frame

video_path = "examples/sample_msrvtt.mp4"

# options ["ShareGPTVideo/LLaVA-Hound-DPO", "ShareGPTVideo/LLaVA-Hound-SFT", "ShareGPTVideo/LLaVA-Hound-SFT-Image_only"]
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

# Inference Example for Detailed Caption Model
To generate detailed video captions with our pretrained ckpt use
```python
import numpy as np
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from inference.inference_utils import ModelInference, decode2frame, detail_templates

video_path = "examples/sample_msrvtt.mp4"

model_path = "ShareGPTVideo/LLaVA-Hound-Pretrain"
model_name = get_model_name_from_path(model_path)
tokenizer, model, processor, context_len = load_pretrained_model(model_path, model_base = None, model_name=model_name, cache_dir=os.environ['CACHE_DIR'])
inference_model = ModelInference(model=model, tokenizer=tokenizer, processor=processor, context_len=context_len)

question = np.random.choice(detail_templates) # use pretrained template questions

# our pipeline
frame_dir, _ = os.path.splitext(video_path)
decode2frame(video_path, frame_dir, verbose=True)
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

Pretrain + SFT refer to [Pretrain + SFT](https://github.com/RifleZhang/LLaVA-Hound-DPO/tree/main/llava_hound_dpo/sft_scripts)

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

# Acknowledgement
Code is build updo the following projects:
- [Video-LLaVA](https://github.com/PKU-YuanGroup/Video-LLaVA) as the LMM architecture
- [trl](https://github.com/huggingface/trl) for DPO implementation

Thanks for their great work!
