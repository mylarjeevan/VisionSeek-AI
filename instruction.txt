# CPU
conda create -n yolo_image_search python=3.11 -y
conda activate yolo_image_search
pip install -r requirements.txt

# GPU
conda create -n yolo_image_search_gpu python=3.11 -y
conda activate yolo_image_search_gpu
conda install pytorch==2.5.1 torchvision==0.20.1 pytorch-cuda=12.4 -c pytorch -c nvidia
pip install -r requirements.txt

# Extra commands
conda remove -n yolo_image_search --all
conda env list
nvidia-smi

# CUDA Installation Guide :
## for-linux :
- https://docs.nvidia.com/cuda/cuda-installation-guide-linux/

## for-windows :
- https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/

## forums-for-more-info :
- https://forums.developer.nvidia.com/t/whats-the-relationship-between-cuda-toolkit-and-pytorch/251096/2