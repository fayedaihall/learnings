#!/bin/bash
# Step 0: Make sure when create the pod: 
# + Add 7860 to exposed HTTP ports
# + Increase Container Disk to 30GB
# + Increase Volume Disk to 100GB

# Step 1: Make sure you're in /workspace (RunPod default writeable disk)
cd /workspace

# Step 2: Update and install system essentials (if needed)
sudo apt-get update

# Step 3: Clone Text Generation WebUI
git clone https://github.com/oobabooga/text-generation-webui.git
cd text-generation-webui

# Step 4: Create and activate a fresh Python virtual environment
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

# Step 5: Install Python requirements
pip install rich
pip install exllamav2
pip install -r requirements/full/requirements.txt 

# Step 6: Ensure CUDA is present and recognized (for GPU pods) [optional]
if [ -d "/usr/local/cuda" ]; then
    echo "CUDA found at /usr/local/cuda"
else
    echo "Warning: CUDA not detected! Make sure you picked a CUDA or ML template pod."
fi

# Step 7: Download model using WebUI's built-in download script
python download-model.py TheBloke/MythoMax-L2-13B-GPTQ

# Step 7 Alternative: Download model manually
mkdir -p models/TheBloke_MythoMax-L2-13B-GPTQ
cd user_data/models/TheBloke_MythoMax-L2-13B-GPTQ
wget https://huggingface.co/TheBloke/MythoMax-L2-13B-GPTQ/resolve/main/model.safetensors
wget https://huggingface.co/TheBloke/MythoMax-L2-13B-GPTQ/resolve/main/config.json
wget https://huggingface.co/TheBloke/MythoMax-L2-13B-GPTQ/resolve/main/tokenizer.json

# Step 8: Start the WebUI server with correct model path, exposed for RunPod HTTP proxy
python server.py --model /workspace/text-generation-webui/user_data/models/TheBloke_MythoMax-L2-13B-GPTQ --api --listen --listen-port 7860

# ---- END ----
