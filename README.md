# LLaVA-Hound-DPO

# setup:
```
source setup/setup_env.sh
```

# inference example

# test
```
# one-line setup data
source setup/setup_test_data.sh

# one-line eval for official (a subset of 5k qa)
bash test/pipeline/outdomain_official_test_pipeline.sh \
$model_output_name \
$model_name

# one-line eval for our in-domain
bash test/pipeline/indomain_test_pipeline.sh \
$model_output_name \
$model_name

# one-line eval for our out-of-domain 
bash test/pipeline/outdomain_test_pipeline.sh \
$model_output_name \
$model_name
```
