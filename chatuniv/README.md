# Chat-UniVi testing with official CKPT

First run one-line set up of video instruction and frames in the [test readme](https://github.com/RifleZhang/LLaVA-Hound-DPO/blob/main/llava_hound_dpo/test/README.md)

# Download Chat-UniVi
```
source set_var.sh
```

# Benchmark Testing
```
# official testing 
bash run_test/pipeline/outdomain_official_test_pipeline.sh

# our indomain testing
bash run_test/pipeline/outdomain_test_pipeline.sh

# our outdomain testing
bash run_test/pipeline/indomain_test_pipeline.sh 
```
