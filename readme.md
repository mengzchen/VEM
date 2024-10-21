conda create -n qwen2vl python==3.8.10
source activate qwen2vl

cd LLaMA-Factory
pip install -e ".[torch,metrics]"

llamafactory-cli train configs/qwen2_vl_lora.yaml
llamafactory-cli export examples/merge_lora/qwen2vl_lora_sft.yaml