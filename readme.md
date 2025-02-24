blob: https://cloudaigcrdataeus.blob.core.windows.net/tunnel-data-code-eus

### environment
conda create -n qwen2vl python==3.8.10
source activate qwen2vl

git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"

### get data

### training critic model
llamafactory-cli train configs/qwen2_vl_lora.yaml
llamafactory-cli export configs/qwen2vl_lora_merge.yaml

### training policy model
use Auto-GUI as policy base model