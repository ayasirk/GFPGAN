FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04

WORKDIR /app

RUN apt-get update
RUN apt-get install -y --no-install-recommends --fix-missing
    # python
RUN apt install -y python3.8 python3-pip python3-setuptools python3-dev
RUN ln -fs /usr/share/zoneinfo/America/New_York /etc/localtime
RUN export DEBIAN_FRONTEND=noninteractive 
RUN apt-get install -y tzdata 
RUN dpkg-reconfigure --frontend noninteractive tzdata
    # OpenCV deps
#RUN apt-get update -y
RUN apt-get install -y libglib2.0-dev libsm6 libxext6 libxrender1 libgl1-mesa-glx g++
    # c++
    # others
RUN apt-get install -y wget
RUN apt-get install unzip

# Ninja
RUN wget https://github.com/ninja-build/ninja/releases/download/v1.8.2/ninja-linux.zip
RUN unzip ninja-linux.zip -d /usr/local/bin/
RUN update-alternatives --install /usr/bin/ninja ninja /usr/local/bin/ninja 1 --force

# basicsr facexlib
RUN python3 -m pip install --upgrade pip 
RUN pip3 install --no-cache-dir torch>=1.7 opencv-python>=4.5
RUN pip3 install --no-cache-dir basicsr facexlib realesrgan

# weights
RUN wget https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth \
        -P experiments/pretrained_models &&\
    wget https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/GFPGANv1.pth \
        -P experiments/pretrained_models

RUN rm -rf /var/cache/apt/* /var/lib/apt/lists/* && \
    apt-get autoremove -y && apt-get clean

COPY . .
RUN pip3 install --no-cache-dir -r requirements.txt

RUN pip3 install .


CMD ["python3", "inference_gfpgan.py", "--upscale", "2", "--test_path", "inputs/whole_imgs", "--save_root", "results"]
