#
# docker network create isa-network
# docker run -d -p 8895:8888 --network isa-network --shm-size=1G -e JUPYTER_ENABLE_LAB=yes --name killianpy -v data:/home/jovyan/data -v notebooks:/home/jovyan/notebooks isischameleon/killian:latest
# docker build -t isischameleon/killian:1 .

#FROM jupyter/datascience-notebook:latest
FROM jupyter/scipy-notebook:latest

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

WORKDIR $HOME/notebooks