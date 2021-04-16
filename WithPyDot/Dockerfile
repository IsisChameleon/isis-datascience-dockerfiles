FROM jupyter/datascience-notebook:latest

USER $NB_UID

RUN mkdir "/home/$NB_USER/data" && \
    mkdir "/home/$NB_USER/notebooks" && \
    fix-permissions "/home/$NB_USER"

# Install Python 3 packages
## node-js: Installing nodejs for frontend debugging
## xeus-python: Installing a kernel supporting Jupyter debugging protocol
RUN conda install -c conda-forge --quiet --yes \
    'python-graphviz' \
    'pydot' \
    'nodejs' \
    'xeus-python' && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

#Installing labextension for frontend debugging
RUN jupyter labextension install @jupyterlab/debugger

WORKDIR $HOME/notebooks

WORKDIR $HOME