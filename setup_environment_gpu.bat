@echo off
conda deactivate && ^
conda create --name action_env python==3.8 -y && ^
conda activate action_env && ^
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html && ^
pip install -r requirements.txt