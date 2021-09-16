# This file may be used to create an environment using:
# $ conda create --name <env> --file <this file>
# platform: linux-64
# ! openblas ! 
# ! CUDA 11.1 !
pip install \
    -f https://pypi.tuna.tsinghua.edu.cn/simple \
    -f https://download.pytorch.org/whl/torch_stable.html \
    easydict==1.9 \
    imageio==2.9.0 \
    numpy==1.21.1 \
    plyfile==0.7.4 \
    tensorboardx==2.2 \
    open3d==0.13.0 \
    torch==1.8.0+cu111

pip install \
    MinkowskiEngine==0.5.4
