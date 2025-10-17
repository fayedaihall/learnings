#!/bin/bash

# Step 1: Make sure you're in /workspace (RunPod default writeable disk)
cd /workspace

# Step 2: Update and install system essentials (if needed)
sudo apt-get update
sudo apt-get install -y git python3 python3-venv

# Step 3: Clone Text Generation WebUI
git clone https://github.com/oobabooga/text-generation-webui.git
cd text-generation-webui

# Step 4: Create and activate a fresh Python virtual environment
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

# Step 5: Install Python requirements
pip install -r requirements.txt

# Step 6: Ensure CUDA is present and recognized (for GPU pods)
if [ -d "/usr/local/cuda" ]; then
    echo "CUDA found at /usr/local/cuda"
else
    echo "Warning: CUDA not detected! Make sure you picked a CUDA or ML template pod."
fi

# Step 7: Download model using WebUI's built-in download script
python download-model.py TheBloke/MythoMax-L2-13B-GPTQ

# Step 8: Start the WebUI server with correct model path, exposed for RunPod HTTP proxy
python server.py --model /workspace/text-generation-webui/user_data/models/TheBloke_MythoMax-L2-13B-GPTQ --api --listen --listen-port 7860

# ---- END ----
