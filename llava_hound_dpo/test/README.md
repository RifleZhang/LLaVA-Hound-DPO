# Test

# Discussion of gpt version and results
In [table 4 Appendix A](https://arxiv.org/pdf/2404.01258.pdf), we found:
1. Different versions of ChatGPT results in different scale of scores
2. By fixing a specific version of ChatGPT, the relative ranking of model remains stable
   
Therefore, we recommend re-run on test set to replicate the result for fair comparison.
Our inference/eval is provide at [rest_result](https://huggingface.co/datasets/ShareGPTVideo/test_video_and_instruction/blob/main/compile_test_result.tar.gz)

**gpt-3.5-turbo-0301 evaluation**
| Methods            | LLM Size | MSVD-QA Acc. | MSVD-QA Score | MSRVTT-QA Acc. | MSRVTT-QA Score | TGIF-QA Acc. | TGIF-QA Score | Summary Avg Acc. | Rank |
|--------------------|----------|--------------|---------------|----------------|-----------------|--------------|---------------|------------------|------|
| Video-ChatGPT (Maaz et al., 2023) | 7B | 78.62 | 4.00 | 71.67 | 3.63 | 56.31 | 3.45 | 68.87 | 6 |
| LLAMA-VID (Li et al., 2023e) | 7B | 82.57 | 4.12 | 71.94 | 3.65 | 59.00 | 3.63 | 71.17 | 4 |
| LLAMA-VID (Li et al., 2023e) | 13B | 83.72 | 4.16 | 73.63 | 3.68 | 59.72 | 3.66 | 72.36 | 3 |
| Chat-UniVi (Jin et al., 2023) | 7B | 80.52 | 4.02 | 66.92 | 3.41 | 57.73 | 3.49 | 68.39 | 7 |
| Video-LLaVA (Lin et al., 2023b) | 7B | 81.44 | 4.08 | 73.29 | 3.65 | 58.34 | 3.61 | 71.02 | 5 |
| LLAVA-HOUND-SFT (ours) | 7B | 85.65 | 4.10 | 73.85 | 3.62 | 64.98 | 3.65 | 74.83 | 2 |
| LLAVA-HOUND-DPO (ours) | 7B | 88.50 | 4.20 | 82.10 | 3.84 | 75.48 | 3.81 | 82.03 | 1 |

**gpt-3.5-turbo-0613 evaluation** 
| Methods            | LLM Size | MSVD-QA Acc. | MSVD-QA Score | MSRVTT-QA Acc. | MSRVTT-QA Score | TGIF-QA Acc. | TGIF-QA Score | Summary Avg Acc. | Rank |
|--------------------|----------|--------------|---------------|----------------|-----------------|--------------|---------------|------------------|------|
| Video-ChatGPT (Maaz et al., 2023) | 7B | 68.55 | 3.80 | 58.90 | 3.36 | 47.83 | 3.21 | 58.43 | 6 |
| LLAMA-VID (Li et al., 2023e) | 7B | 72.62 | 3.92 | 58.73 | 3.38 | 49.21 | 3.28 | 60.19 | 4 |
| LLAMA-VID (Li et al., 2023e) | 13B | 74.29 | 3.96 | 59.82 | 3.41 | 50.83 | 3.33 | 61.65 | 3 |
| Chat-UniVi (Jin et al., 2023) | 7B | 70.01 | 3.79 | 53.08 | 3.14 | 46.09 | 3.12 | 56.39 | 7 |
| Video-LLaVA (Lin et al., 2023b) | 7B | 71.75 | 3.88 | 58.97 | 3.39 | 48.39 | 3.24 | 59.70 | 5 |
| LLAVA-HOUND-SFT (ours) | 7B | 75.70 | 3.86 | 58.73 | 3.31 | 53.51 | 3.30 | 62.65 | 2 |
| LLAVA-HOUND-DPO (ours) | 7B | 80.73 | 4.07 | 70.15 | 3.66 | 61.38 | 3.46 | 70.75 | 1 |

**gpt-3.5-turbo-1106 evaluation**
| Methods            | LLM Size | MSVD-QA Acc. | MSVD-QA Score | MSRVTT-QA Acc. | MSRVTT-QA Score | TGIF-QA Acc. | TGIF-QA Score | Summary Avg Acc. | Rank |
|--------------------|----------|--------------|---------------|----------------|-----------------|--------------|---------------|------------------|------|
| Video-ChatGPT (Maaz et al., 2023) | 7B | 73.02 | 4.01 | 62.09 | 3.61 | 47.76 | 3.36 | 60.96 | 6 |
| LLAMA-VID (Li et al., 2023e) | 7B | 75.49 | 4.08 | 62.09 | 3.61 | 51.72 | 3.47 | 63.10 | 4 |
| LLAMA-VID (Li et al., 2023e) | 13B | 76.97 | 4.10 | 63.16 | 3.61 | 52.53 | 3.50 | 64.22 | 3 |
| Chat-UniVi (Jin et al., 2023) | 7B | 72.22 | 3.92 | 55.06 | 3.35 | 48.16 | 3.31 | 58.47 | 7 |
| Video-LLaVA (Lin et al., 2023b) | 7B | 74.76 | 4.04 | 62.70 | 3.60 | 51.24 | 3.45 | 62.89 | 5 |
| LLAVA-HOUND-SFT (ours) | 7B | 81.09 | 4.08 | 64.13 | 3.57 | 58.05 | 3.53 | 67.76 | 2 |
| LLAVA-HOUND-DPO (ours) | 7B | 86.05 | 4.23 | 76.75 | 3.85 | 70.02 | 3.71 | 77.61 | 1 |


# One-line cmd Video Instruction and Video Frames Setup
Follow environment set up in [main page setup](https://github.com/RifleZhang/LLaVA-Hound-DPO/blob/main/README.md#setup)
Download data
```
source setup/setup_test_data.sh
```

# One-line cmd for Benchmark Testing 
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
2. One-line testing for [LLama-Vid](https://github.com/RifleZhang/LLaVA-Hound-DPO/blob/main/llama_vid/README.md):

  Reference: [LLaMA-VID: An Image is Worth 2 Tokens in Large Language Models](https://arxiv.org/abs/2311.17043)
  
3. One-line testing for [Chat-UniVi](https://github.com/RifleZhang/LLaVA-Hound-DPO/tree/main/chatuniv): 

  Reference: [Chat-UniVi: Unified Visual Representation Empowers Large Language Models with Image and Video Understanding](https://arxiv.org/abs/2311.08046)


# Customized Model Testing Instruction
Requirement: 
1. pretrained model ckpt, better a [huggingface model card](https://huggingface.co/docs/hub/en/model-cards)
2. inference function takes video or frames as input.

Only two parts need to be implemented:
1. [model_loading_function](https://github.com/RifleZhang/LLaVA-Hound-DPO/blob/main/chatuniv/run_test/inference/inference_test_qa.py#L62)
2. [Inference function](https://github.com/RifleZhang/LLaVA-Hound-DPO/blob/main/chatuniv/chatuniv_utils.py#L120) 



