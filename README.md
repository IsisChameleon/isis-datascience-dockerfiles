# isis-datascience-dockerfiles

This docker file builds a data science container with Jupyter based on the Jupyter stack Data Science image, including Python R and Julia and a few libraries.

Additionnaly, it installs the JupyterLab visual debugger and a few other libraries (python Graphviz)

More details on how to use and run the container here:

https://hub.docker.com/repository/docker/isischameleon/usyd-data-science

docker build --t isischameleon/usyd-data-science:debugger .

# to enable GPU when doing docker run: --gpus all
# https://medium.com/swlh/still-wondering-how-to-set-up-a-docker-interpreter-with-pycharm-5bfdb8e1e65d
docker run 
-it #Make docker session interactive
--rm #Automatically remove the container when it exits
--gpus all #Enable GPU support in the container
-e DISPLAY=${DISPLAY} #Pass environment variable display
--net=host #Share network configu of the host with this container
--user "$(id -u):$(id -g)" #Pass user id and group id
-v <path to pycharm in your host machine>:<path to pycharm in docker container> #Volume mounting to access pycharm from the container
myimage
