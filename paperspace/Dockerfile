#
# docker network create isa-network
# docker run -d -p 8895:8888 --network isa-network --shm-size=8G -e JUPYTER_ENABLE_LAB=yes --name scipytorch -v data:/home/jovyan/data -v notebooks:/home/jovyan/notebooks isischameleon/usyd-pytorch:3
# docker run -d -p 8895:8888 --network isa-network --shm-size=8G -e JUPYTER_ENABLE_LAB=yes --name scipytorch --mount type=bind,source="F:\Data\Datasets",target=/home/jovyan/data -v notebooks:/home/jovyan/notebooks isischameleon/usyd-pytorch:3
# docker build -t isischameleon/paperspace-pyg:1 .

# https://peteris.rocks/blog/quiet-and-unattended-installation-with-apt-get/

#FROM jupyter/datascience-notebook:latest
FROM paperspace/nb-pytorch:22.02-py3

USER root

RUN printf 'tzdata  tzdata/Areas    select  Australia \ntzdata  tzdata/Zones/Australia  select  Sydney \
\ndebconf debconf/frontend        select  Noninteractive\nkeyboard-configuration  keyboard-configuration/layout   select  English (Australian) ' >> seed.txt

RUN debconf-set-selections seed.txt

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin && \
    mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600

RUN wget https://developer.download.nvidia.com/compute/cuda/11.5.0/local_installers/cuda-repo-ubuntu1804-11-5-local_11.5.0-495.29.05-1_amd64.deb && \
    dpkg -i cuda-repo-ubuntu1804-11-5-local_11.5.0-495.29.05-1_amd64.deb && \
    apt-key add /var/cuda-repo-ubuntu1804-11-5-local/7fa2af80.pub && \
    apt-get update --yes

RUN apt-get -qq install cuda || true && \
    rm   /usr/local/cuda && \
    ln -s /usr/local/cuda-11.5 /usr/local/cuda

RUN conda install --quiet --yes pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.5 -c pytorch -c conda-forge

RUN pip install torch-scatter torch-sparse torch-geometric -f https://data.pyg.org/whl/torch-1.11.0+cu115.html

WORKDIR /notebooks