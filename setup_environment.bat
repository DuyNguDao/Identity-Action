@echo off
conda deactivate && ^
conda create --name face_env python==3.8 -y && ^
conda activate action_env && ^
pip install -r requirements.txt