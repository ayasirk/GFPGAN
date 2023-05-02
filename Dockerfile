FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04

WORKDIR /app

RUN apt-get update
#RUN apt-get install -y --no-install-recommends --fix-missing \
    # python
RUN apt install python3.8
RUN apt-get install python3-pip
RUN apt-get install python3-setuptools 
RUN apt-get install python3-dev 
    # OpenCV deps
RUN apt-get install libglib2.0-0 
RUN apt-get install libsm6
RUN apt-get install libxext6
RUN apt-get install libxrender1
RUN apt-get install libgl1-mesa-glx
    # c++
RUN apt-get install g++ 
    # others
RUN    wget unzip

# Ninja
RUN wget https://github.com/ninja-build/ninja/releases/download/v1.8.2/ninja-linux.zip && \
    unzip ninja-linux.zip -d /usr/local/bin/ && \
    update-alternatives --install /usr/bin/ninja ninja /usr/local/bin/ninja 1 --force

# basicsr facexlib
RUN python3 -m pip install --upgrade pip && \
    pip3 install --no-cache-dir torch>=1.7 opencv-python>=4.5 && \
    pip3 install --no-cache-dir basicsr facexlib realesrgan

# weights
RUN wget https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth \
        -P experiments/pretrained_models &&\
    wget https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/GFPGANv1.pth \
        -P experiments/pretrained_models

RUN rm -rf /var/cache/apt/* /var/lib/apt/lists/* && \
    apt-get autoremove -y && apt-get clean

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .
RUN pip3 install .


CMD ["python3", "inference_gfpgan.py", "--upscale", "2", "--test_path", "inputs/whole_imgs", "--save_root", "results"]
