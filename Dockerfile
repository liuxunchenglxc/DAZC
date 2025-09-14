FROM nvidia/cuda:11.4.3-cudnn8-devel-ubuntu20.04

RUN apt update && \
apt install -y python3-pip && \
DEBIAN_FRONTEND=noninteractive apt-get install -y ffmpeg libsm6 libxext6 libglib2.0-0 libxrender-dev && \
ln -s /usr/bin/python3 /usr/bin/python && \
apt-get clean && \
rm -rf /var/lib/apt/lists/*

RUN pip install torch==2.0.1 timm scikit-image ptflops easydict PyYAML pillow torchvision opencv-python

RUN pip install scipy

CMD ["/bin/bash"]