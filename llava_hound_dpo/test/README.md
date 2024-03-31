# Test

# TODO: Add discussion of gpt version and results

# One-line cmd Video Instruction and Video Frames Setup
```
source setup/setup_test_data.sh
```

# One-line cmd for Benmark Testing 
Eval for exisiting benchmark dataset from [Video-ChatGPT](https://github.com/mbzuai-oryx/Video-ChatGPT?tab=readme-ov-file#quantitative-evaluation-bar_chart) benchmark Video QA dataset
Our is a subset test sample about 5k for each datasets, but the variance is verified to be within ~0.3 accuracy from the full dataset.
```
bash test/pipeline/outdomain_official_test_pipeline.sh \
$model_output_name \
$model_name
```

Exampe of testing with LLaVA-Hound-DPO model
```
bash test/pipeline/indomain_official_test_pipeline.sh \
llava_hound_dpo \
ShareGPTVideo/LLaVA-Hound-DPO
```

Exampe of testing official Video-LLaVA model
```
bash test/pipeline/indomain_official_test_pipeline.sh \
videollava \
LanguageBind/Video-LLaVA-7B
```

# Evaluation on Our In-domain Video QA Benchmark
```
bash test/pipeline/indomain_test_pipeline.sh \
$model_output_name \
$model_name
```
Example
```
bash test/pipeline/indomain_test_pipeline.sh \
llava_hound_dpo \
ShareGPTVideo/LLaVA-Hound-DPO
```

# Evaluation on Our Out-of-domain Video QA Benchmark
```
bash test/pipeline/outdomain_test_pipeline.sh \
$model_output_name \
$model_name
```

Example
```
bash test/pipeline/outdomain_test_pipeline.sh \
llava_hound_dpo \
ShareGPTVideo/LLaVA-Hound-DPO
```

# Other SOTA model testing
1. One-line testing for [Video-ChatGPT](https://github.com/RifleZhang/LLaVA-Hound-DPO/blob/main/video_chatgpt/README.md)

   Reference: [Video-ChatGPT: Towards Detailed Video Understanding via Large Vision and Language Models](https://arxiv.org/abs/2306.05424)
2. One-line testing for LLama-Vid:

  Reference: [LLaMA-VID: An Image is Worth 2 Tokens in Large Language Models](https://arxiv.org/abs/2311.17043)
3. One-line testing for ChatUniV: 

  Reference: [Chat-UniVi: Unified Visual Representation Empowers Large Language Models with Image and Video Understanding](https://arxiv.org/abs/2311.08046)

