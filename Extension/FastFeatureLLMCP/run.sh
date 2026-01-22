
###  Qwen-72B  ####

python uncertainty_quantification_via_ffcp.py \
  --model=Qwen-72B \
  --raw_data_dir=data \
  --logits_data_dir=outputs_base \
  --cal_ratio=0.5 \
  --alpha=0.1 

python uncertainty_quantification_via_fcp.py \
  --model=Qwen-72B \
  --raw_data_dir=data \
  --logits_data_dir=outputs_base \
  --cal_ratio=0.5 \
  --alpha=0.1 

python uncertainty_quantification_via_cp.py \
  --model=Qwen-72B \
  --raw_data_dir=data \
  --logits_data_dir=outputs_base \
  --cal_ratio=0.5 \
  --alpha=0.1 

###  Qwen-14B  ####

python uncertainty_quantification_via_ffcp.py \
  --model=Qwen-14B \
  --raw_data_dir=data \
  --logits_data_dir=outputs_base \
  --cal_ratio=0.5 \
  --alpha=0.1 

python uncertainty_quantification_via_fcp.py \
  --model=Qwen-14B \
  --raw_data_dir=data \
  --logits_data_dir=outputs_base \
  --cal_ratio=0.5 \
  --alpha=0.1 

python uncertainty_quantification_via_cp.py \
  --model=Qwen-14B \
  --raw_data_dir=data \
  --logits_data_dir=outputs_base \
  --cal_ratio=0.5 \
  --alpha=0.1 

####  Qwen-7B  ####

python uncertainty_quantification_via_ffcp.py \
  --model=Qwen-7B \
  --raw_data_dir=data \
  --logits_data_dir=outputs_base \
  --cal_ratio=0.5 \
  --alpha=0.1 

python uncertainty_quantification_via_fcp.py \
  --model=Qwen-7B \
  --raw_data_dir=data \
  --logits_data_dir=outputs_base \
  --cal_ratio=0.5 \
  --alpha=0.1 

python uncertainty_quantification_via_cp.py \
  --model=Qwen-7B \
  --raw_data_dir=data \
  --logits_data_dir=outputs_base \
  --cal_ratio=0.5 \
  --alpha=0.1 

####  Qwen-1.8B  ####

python uncertainty_quantification_via_ffcp.py \
  --model=Qwen-1_8B \
  --raw_data_dir=data \
  --logits_data_dir=outputs_base \
  --cal_ratio=0.5 \
  --alpha=0.1 

python uncertainty_quantification_via_fcp.py \
  --model=Qwen-1_8B \
  --raw_data_dir=data \
  --logits_data_dir=outputs_base \
  --cal_ratio=0.5 \
  --alpha=0.1 

python uncertainty_quantification_via_cp.py \
  --model=Qwen-1_8B \
  --raw_data_dir=data \
  --logits_data_dir=outputs_base \
  --cal_ratio=0.5 \
  --alpha=0.1 

###  deepseek-llm-7b  ####

python uncertainty_quantification_via_ffcp.py \
  --model=deepseek-llm-7b-base \
  --raw_data_dir=data \
  --logits_data_dir=outputs_base \
  --cal_ratio=0.5 \
  --alpha=0.1 

python uncertainty_quantification_via_fcp.py \
  --model=deepseek-llm-7b-base \
  --raw_data_dir=data \
  --logits_data_dir=outputs_base \
  --cal_ratio=0.5 \
  --alpha=0.1 \
  --delta=0.4

python uncertainty_quantification_via_cp.py \
  --model=deepseek-llm-7b-base \
  --raw_data_dir=data \
  --logits_data_dir=outputs_base \
  --cal_ratio=0.5 \
  --alpha=0.1 

####  deepseek-llm-67b  ####

python uncertainty_quantification_via_ffcp.py \
  --model=deepseek-llm-67b-base \
  --raw_data_dir=data \
  --logits_data_dir=outputs_base \
  --cal_ratio=0.5 \
  --alpha=0.1 

python uncertainty_quantification_via_fcp.py \
  --model=deepseek-llm-67b-base \
  --raw_data_dir=data \
  --logits_data_dir=outputs_base \
  --cal_ratio=0.5 \
  --alpha=0.1 \
  --delta=0.4

python uncertainty_quantification_via_cp.py \
  --model=deepseek-llm-67b-base \
  --raw_data_dir=data \
  --logits_data_dir=outputs_base \
  --cal_ratio=0.5 \
  --alpha=0.1 