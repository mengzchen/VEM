conda create -n qwen2vl python==3.8.10
conda activate qwen2vl

pip install git+https://github.com/huggingface/transformers

cd LLaMA-Factory
pip install -e ".[torch,metrics]"
pip install --no-deps -e .
pip install deepspeed
pip install flash-attn --no-build-isolation

llamafactory-cli train configs/qwen2_vl_lora.yaml
llamafactory-cli export examples/merge_lora/qwen2vl_lora_sft.yaml