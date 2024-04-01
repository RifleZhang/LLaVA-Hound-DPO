# LLaVA-Hound-DPO

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
