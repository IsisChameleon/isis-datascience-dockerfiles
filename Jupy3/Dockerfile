#
# docker network create isa-network
# docker run -d -p 8895:8888 --network isa-network --shm-size=8G -e JUPYTER_ENABLE_LAB=yes --name scipytorch -v data:/home/jovyan/data -v notebooks:/home/jovyan/notebooks isischameleon/usyd-pytorch:3
# docker run -d -p 8895:8888 --network isa-network --shm-size=8G -e JUPYTER_ENABLE_LAB=yes --name scipytorch --mount type=bind,source="F:\Data\Datasets",target=/home/jovyan/data -v notebooks:/home/jovyan/notebooks isischameleon/usyd-pytorch:3
# docker run -d -p 8890:8888 --network isa-network --shm-size=8G -e JUPYTER_ENABLE_LAB=yes --name scipytorch --mount type=bind,source="F:\Data\Datasets",target=/home/jovyan/data --mount type=bind,source="D:\src\work",target=/home/jovyan/notebooks isischameleon/usyd-pytorch:3
# docker build -t isischameleon/usyd-pytorch:3 .


# to enable GPU when doing docker run: --gpus all
# https://medium.com/swlh/still-wondering-how-to-set-up-a-docker-interpreter-with-pycharm-5bfdb8e1e65d
# docker run 
# -it #Make docker session interactive
# --rm #Automatically remove the container when it exits
# --gpus all #Enable GPU support in the container
# -e DISPLAY=${DISPLAY} #Pass environment variable display
# --net=host #Share network configu of the host with this container
# --user "$(id -u):$(id -g)" #Pass user id and group id
# -v <path to pycharm in your host machine>:<path to pycharm in docker container> #Volume mounting to access pycharm from the container
# myimage

#FROM jupyter/datascience-notebook:latest
FROM jupyter/scipy-notebook:latest

USER root

RUN apt-get update --yes && \
    apt-get install --yes --no-install-recommends \
    default-jdk \
    ant

USER $NB_UID

RUN mkdir "/home/$NB_USER/data" && \
    mkdir "/home/$NB_USER/notebooks" && \
    fix-permissions "/home/$NB_USER"

# Install Python 3 packages
## node-js: Installing nodejs for frontend debugging
## xeus-python: Installing a kernel supporting Jupyter debugging protocol
RUN conda install -c conda-forge --quiet --yes \
    'python-graphviz' \
    'jupyterlab-git' \
    'nodejs' \
    'jpype1' \
    'networkx' \
    'xeus-python' && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

RUN jupyter labextension install @jupyterlab/debugger @jupyterlab/git

RUN conda install -c pytorch --quiet --yes \
    'pytorch' \
    'torchvision' && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

RUN torchversion = torch.__version__

WORKDIR $HOME/notebooks