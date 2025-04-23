# [NaPa] Code Implementation

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Official code for the paper:  
**"NaPa: Novel 3D-Aware Composition Images Synthesis for Product Display with Diffusion Model**  
*Tao Xu, Lianghong Li, Xiaoshuai Zhang, Liangyou Li, Ying Zang, Liangyou Li, Lianghong Li, Da Fang*  


---

### Installation

#### Step 1: Install One-2-3-45
<details>
<summary>Traditional Installation</summary>

```bash
# System packages
sudo apt update && sudo apt install git-lfs libsparsehash-dev build-essential

# Conda setup
conda create -n NaPa python=3.10
conda activate NaPa

cd One-2-3-45

# Install dependencies
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run
export PATH="/usr/local/cuda-11.8/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH"

pip install torch==2.0.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# Build special dependencies
export TORCH_CUDA_ARCH_LIST="7.0;7.2;8.0;8.6+PTX"
export IABN_FORCE_CUDA=1
pip install inplace_abn
FORCE_CUDA=1 pip install git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0

# Download models
python download_ckpt.py
### Installation by Docker Images
<details>
<summary>Option 1: Pull and Play (environment and checkpoints). (~22.3G)</summary> 

```bash
# Pull the Docker image that contains the full repository.
docker pull chaoxu98/one2345:demo_1.0
# An interactive demo will be launched automatically upon running the container.
# This will provide a public URL like XXXXXXX.gradio.live
docker run --name One-2-3-45_demo --gpus all -it chaoxu98/one2345:demo_1.0
```
</details>

<details>
<summary>Option 2: Environment Only. (~7.3G)</summary> 

```bash
# Pull the Docker image that installed all project dependencies.
docker pull chaoxu98/one2345:1.0
# Start a Docker container named One2345.
docker run --name One-2-3-45 --gpus all -it chaoxu98/one2345:1.0
# Get a bash shell in the container.
docker exec -it One-2-3-45 /bin/bash
# Clone the repository to the local machine.
git clone https://github.com/One-2-3-45/One-2-3-45
cd One-2-3-45
# Download model checkpoints. 
python download_ckpt.py
# Refer to getting started for inference.
```
</details>

#### Step 2: Forgery Detection (Optional)

### Setup
```bash
pip install -r Explicit-Visual-Prompt/requirements.txt
```

### Usage
#### Step 1: Run One-2-3-45 3D Lifting
```bash
python run.py --img_path INPUT_IMG --half_precision
```
#### Step 2: Perform Image Synthesis
```bash
python NaPa.py\
    --path ./input_images \
    --prompts "Spaceship over the white screen.>Spaceship in the city.,Spaceship over the white screen.>Spaceship in the sky." \
    --output ./results \
    --seed 123456789
```
#### Step 3: Forgery Detection
```bash
python Explicit-Visual-Prompt/forgery_detection.py --input_path  
```
