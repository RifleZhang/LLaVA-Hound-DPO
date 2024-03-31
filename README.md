# LLaVA-Hound-DPO

# setup:
```
source setup/setup_env.sh
```

# inference example

# Testing with one-line command 
```
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
```
bash test/pipeline/outdomain_official_test_pipeline.sh \
videollava_dpo \
ShareGPTVideo/LLaVA-Hound-DPO
```
