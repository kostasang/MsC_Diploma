FROM nvcr.io/nvidia/l4t-base:r32.2
#FROM nvcr.io/nvidia/l4t-base:r32.6.1

WORKDIR /workdir

RUN wget https://nvidia.box.com/shared/static/ncgzus5o23uck9i5oth2n8n06k340l6k.whl --no-check-certificate -O  torch-1.4.0-cp36-cp36m-linux_aarch64.whl
#RUN wget https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl --no-check-certificate -O  torch-1.10.0-cp36-cp36m-linux_aarch64.whl 

RUN apt-get update && apt-get install -y --no-install-recommends apt-utils

RUN apt-get install -y --no-install-recommends python3-pip python3-venv python3-setuptools python3-dev libpython3-dev 
RUN apt-get install -y --no-install-recommends libopenblas-base libopenmpi-dev libomp-dev liblapack-dev libatlas-base-dev gfortran

RUN pip3 install Cython
RUN pip3 install --upgrade pip
RUN pip3 install --upgrade protobuf
RUN pip3 install numpy==1.19.4
RUN pip3 install torch-1.4.0-cp36-cp36m-linux_aarch64.whl
#RUN pip3 install torch-1.10.0-cp36-cp36m-linux_aarch64.whl 
