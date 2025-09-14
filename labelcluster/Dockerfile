FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

RUN apt update && \
apt install -y python3-pip && \
DEBIAN_FRONTEND=noninteractive apt-get install -y ffmpeg libsm6 libxext6 libglib2.0-0 libxrender-dev && \
ln -s /usr/bin/python3 /usr/bin/python && \
apt-get clean && \
rm -rf /var/lib/apt/lists/*

RUN pip install torch==2.0.1 transformers

RUN pip install kmeans-pytorch

CMD ["/bin/bash"]