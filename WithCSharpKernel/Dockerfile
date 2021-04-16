FROM mcr.microsoft.com/dotnet/sdk As Stage0

FROM jupyter/datascience-notebook:latest As Stage1

COPY --from=Stage0 . .

USER $NB_UID

# RUN apt-get update \
#     && apt-get install -y --no-install-recommends \
#         apt-transport-https \
#     && apt-get update \
#     && apt-get install -y --no-install-recommends \
#         dotnet-sdk 

RUN mkdir "/home/$NB_USER/data" && \
    mkdir "/home/$NB_USER/notebooks" && \
    fix-permissions "/home/$NB_USER"

# Install Python 3 packages
## node-js: Installing nodejs for frontend debugging
## xeus-python: Installing a kernel supporting Jupyter debugging protocol
RUN conda install -c conda-forge --quiet --yes \
    'python-graphviz' \
    'nodejs' \
    'xeus-python' && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

#Installing labextension for frontend debugging 
RUN jupyter labextension install @jupyterlab/debugger

WORKDIR $HOME/notebooks
