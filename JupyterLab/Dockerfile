#
# docker run -d -p 8890:8888 --name tensy -v data:/home/jovyan/data -v notebooks:/home/jovyan/notebooks isischameleon/usyd-tensorflow:latest start.sh jupyterlab
# docker build -t isischameleon/usyd-tensorflow:latest .

#FROM jupyter/datascience-notebook:latest
FROM jupyter/tensorflow-notebook:latest

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
    'xeus-python' && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

#Installing labextension for frontend debugging
RUN jupyter labextension install @jupyterlab/debugger @jupyterlab/git
# RUN conda install -c conda-forge --quiet --yes \
#     'jupyterlab-git'
# RUN conda install -c conda-forge --quiet --yes \
#     'python-graphviz'
# RUN conda install -c conda-forge --quiet --yes \
#     'nodejs'
# RUN conda install -c conda-forge --quiet --yes \
#     'pylint' 
# RUN conda install -c conda-forge --quiet --yes \
#     'xeus-python'
# RUN conda install -c conda-forge --quiet --yes \
#     'nltk' && \
#     fix-permissions "${CONDA_DIR}" && \
#     fix-permissions "/home/${NB_USER}"

#Installing labextension for frontend debugging
# RUN jupyter labextension install @jupyterlab/debugger @jupyterlab/jupyterlab-git


WORKDIR $HOME/notebooks