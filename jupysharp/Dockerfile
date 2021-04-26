################################################################################################
# docker build . -t isischameleon/jupysharp:first
#
# Get the jupyter/base-notebook image (or another suitable image from jupyter stacks)
# update the image for;
# PowerShell 7
# Dot Net Core 3.1 SDK
# Dot Net Interactive
# 
# docker run -d -p 8892:8888 --name tensysharp -v notebooks:/home/joyvan/notebooks isischameleon/jupysharp:first start.sh jupyter lab
# 
################################################################################################

FROM jupyter/tensorflow-notebook:latest

WORKDIR ${HOME}

USER root

# https://docs.microsoft.com/en-us/dotnet/core/install/linux-ubuntu#2004-
# .NET core SDK 3.1.14
# Powershell 7



RUN wget https://packages.microsoft.com/config/ubuntu/20.04/packages-microsoft-prod.deb -O packages-microsoft-prod.deb
RUN dpkg -i packages-microsoft-prod.deb

RUN apt-get update && \
  apt-get install -y apt-transport-https software-properties-common && \
  apt-get update && \
  apt-get install -y dotnet-sdk-5.0

RUN apt-get update && \ 
    add-apt-repository universe && \
    apt-get install -y powershell

USER $NB_USER

RUN dotnet tool install --global Microsoft.dotnet-interactive --version 1.0.221505
ENV PATH="/${HOME}/.dotnet/tools:${PATH}"
ENV DOTNET_CLI_TELEMETRY_OPTOUT=1

RUN dotnet interactive jupyter install


# RUN mkdir /install
# RUN chown -R $NB_USER /install
# WORKDIR /install

# COPY ./dotnet-install.sh .

# RUN ./dotnet-install.sh


# USER root

# ENV MONO_VERSION 5.0.1.1

# RUN apt-get update \
#   && apt-get install -y \ 
#   curl \
#   git \
#   && rm -rf /var/lib/apt/lists/*

# RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 3FA7E0328081BFF6A14DA29AA6A19B38D3D831EF

# RUN echo "deb http://download.mono-project.com/repo/debian jessie/snapshots/$MONO_VERSION main" > /etc/apt/sources.list.d/mono-official.list \
#   && apt-get update \
#   && apt-get install -y binutils mono-devel ca-certificates-mono fsharp mono-vbnc nuget referenceassemblies-pcl \
#   && rm -rf /var/lib/apt/lists/* /tmp/*

# RUN cert-sync /etc/ssl/certs/ca-certificates.crt \
#   && yes | certmgr -ssl -m https://go.microsoft.com \
#   && yes | certmgr -ssl -m https://nugetgallery.blob.core.windows.net \
#   && yes | certmgr -ssl -m https://nuget.org

# RUN mkdir /icsharp
# RUN chown -R $NB_USER /icsharp
# WORKDIR /icsharp

# RUN chown -R $NB_USER $HOME/.config/

USER $NB_USER



# Install .NET kernel
# RUN dotnet tool install -g --add-source "https://dotnet.myget.org/F/dotnet-try/api/v3/index.json" Microsoft.dotnet-interactive
# ENV PATH="/${HOME}/.dotnet/tools:${PATH}"
# ENV DOTNET_CLI_TELEMETRY_OPTOUT=1
# RUN dotnet interactive jupyter install


USER $NB_UID

RUN mkdir "/home/$NB_USER/data" && \
    mkdir "/home/$NB_USER/notebooks" && \
    fix-permissions "/home/$NB_USER"

USER root

RUN chown -R root $HOME/.config/

# Switch back to jovyan to avoid accidental container runs as root
USER $NB_USER