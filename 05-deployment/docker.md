# Docker

main resources:

- [x] [youtube of nana](https://www.youtube.com/watch?v=3c-iBn73dDE&t=1756s)
- [x] [Machine Learning bookcamp](https://www.manning.com/books/machine-learning-bookcamp)
- [ ] [youtube of data-engineering zoomcamp](https://www.youtube.com/watch?v=EYNwNlOrpr0&list=PL3MmuxUbc_hJed7dXYoJw8DoCuVHhGEQb)
- [ ] [youtube of jadi](https://www.youtube.com/watch?v=_jKNnHROiC0&t=806s)
- [ ] [dockerSwarm.pdf](../images/dockerSwarm.pdf)
- [x] [Kubernetes in Action](https://www.manning.com/books/kubernetes-in-action)

## Docker Basics

**what is Container**: a portable package of applications with all their dependencies and configurations which makes
development easier.

container is a stack of images on top of each other. The OS (linux) image is at the button and the application image is
at the top.

![Untitled](../images/container.png)

container repositories like [docker hub](https://hub.docker.com/) stores a lot of container images.

---

* get the image from docker hub

```commandline
docker pull imageName:version
```

after running this, all layers of the container will be downloaded

![Untitled](../images/layersOfImages.png)

If layer exists locally, it will not get downloaded again (can occur in update image which only different layers will be
downloaded)

---

* creates a container from image, starts the container, if not found locally get the image from docker hub
  ![dockerRun.png](../images/dockerRun.png)
  ![dockerRun2.png](../images/dockerRun2.png)

```commandline
docker run imageName:version
docker run -d imageName
docker run --name containerName imageName:version
```

the `-d` runs the container in the detached mode

---

* see all running containers

```commandline
docker ps
docker ps -a
```

the `-a` will show all the running and stopped containers

### docker vs VMs

OS is made up of 2 layers, kernel and application.
The **main difference between docker and VMs** is that **docker virtualizes application layer and uses host's kernel**
while **VMs virtualize both kernel and application layer**.

![Untitled](../images/dockerVsVMs.png)

therefore:

1. size of docker is smaller than VM
2. docker containers are faster
3. linux based docker images are not compatible with the Windows kernel (should use docker toolbox for mac and Windows)

### container vs image

container is a running env for image

container consists of application image + file system + env configs

---

* see all existing images

```commandline
docker images
```

---

* start, stop and restart the container

```commandline
docker start containerId
docker stop containerId
docker restard containerId
```

`docker run` creates a container while `docker start` restarts a stopped container

have in mind that when you restart the container, data will be lost and every configuration will be gone. Data is not
persistent in containers unless you use **Volumes**.

`docker run busybox echo "Hello world` the image is `busybox` and the command is `echo "hello world"`. these two are different things. commands run in the container not the image.

### container port vs host port

![Untitled](../images/containerAndHostPorts.png)
in order to connect to a container, you should call the host port which is mapped to the container port

* run a container with binding host port to container port

```commandline
docker run -p hostPort:containerPort imageName
```

* see the logs of a container

```commandline
docker logs containerId
docker logs containerName | tail
```

`tail` is for showing the last logs

* **go to the terminal of the running container**

```commandline
docker exec -it containerId bash
docker exec -it containerName /bin/bash
docker exec -it containerName /bin/sh -u userName
```

which `-it` means iterative terminal and `-u` means to enter the terminal of which user

some containers don't have bash and should connect to shell instead

---

### docker network

all containers in a docker network can see each other using container name.
all containers outside a docker network should call the container with its domain and port.

* see all docker networks

```commandline
docker network ls
```

* create a docker network

```commandline
docker network create networkName
```

* run a container in a docker network

```commandline
docker run --net networkName imageName
```

---

#### docker run command with all it's options:

```commandline
docker run -d \
    --network networkName \
    --name containerName \
    -p hostPort:containerPort \
    -e INVIRONMENTAL_VARIABLE_NAME = envVariableValue \
    imageName:tag
```

* `-d` is for detached mode
* `-e` is for environmental variables
* `-p` is for port

by running `env` inside the container (`docker exec -it containrId bin/bash`) you can see all the env variables.

---

## Dockerfile

Dockerfile is a blueprint for building images.
![Untitled](../images/Dockerfile.png)
![Untitled](../images/DockerfileAndCommands.png)

* `FROM` Defines the base image to use and start the build process. means that the dockerfile is going to start by
  basing it on what image
  ![Untitled](../images/imageLayers.png)
    * each image is based on another image. image layers are all the images which are based on each other to build the
      final image
* `ENV` you can set env variables on dockerfile too, but it's preferred to have them on docker compose file
* `RUN` you can execute any linux command inside the container
* `COPY` copy files from the host machine to the container and is different with `RUN cp` which runs on container
* `WORKDIR` set up the working directory inside the container. This is roughly equivalent to the `cd` command in Linux (
  change directory), so everything we will run after that will be executed in the mentioned folder.
* `EXPOSE` specify which port our application will use.
* `CMD` or `ENTRYPOINT` command (you can have multiple `RUN`commands but only one `CMD`) which defines what
  should be run when running the image `docker run -it image:tag`. It tells how our application should be started

example Dockerfile

```dockerfile
# base Docker image that we will build on
FROM python:3.9.1

# set up our image by installing prerequisites; pandas in this case
RUN pip install pandas

# set up the working directory inside the container
WORKDIR /app
# copy the script to the container. 1st name is source file, 2nd is destination
COPY pipeline.py pipeline.py

# define what to do first when the container runs
# in this example, we will just run the script
ENTRYPOINT ["python", "pipeline.py"]
```

- each line in the dockerfile is a new layer on top of each other
- docker daemon builds the images
- docker client uploads the data (files, dockerfile) to the daemon
- `ENTRYPOINT` does not execute during the image **build** process. The `ENTRYPOINT` is only executed when a **container** is created and run from the image.

---

* create an image from Dockerfile

```commandline
docker build -t imageName:tagName DockerfilePath
```

`-t` is for tag name of the image

and then you can run your image with `docker run imageName:tagName`
> what Jenkins do is that it create an image from the docker file and push the image to the docker repository.
>
> the image then can be pulled from docker repository to be run on local for development, test env or production

in order to rebuild an image (when you changed the docker file), you have to delete to old image

* delete a container

```commandline
docker rm containerId
```

* remove an image

```commandline
docker rmi imageId
```

you can only delete an image when no container is using it.

---

## docker compose

instead of running multiple containers in terminal, docker compose automate the process by writing all configurations in
a **YAML** file.
![Untitled](../images/dockerCompose.png)

example docker compose:

```yml
version: "3.9"
services:
  clothing-model:
    image: zoomcamp-10-model:xception-v4-001
  gateway:
    image: zoomcamp-10-gateway:002
    environment:
      - TF_SERVING_HOST=clothing-model:8500
    ports:
      - "9696:9696"
```

- as you can see you can have multiple services with different images in one docker compose
- docker compose create a common network for all its containers. so all the services can see each other
- configs of each image is preferred to define in the Dockerfile. However, configs related to communication of these
  services (network stuff) is preferred to be defined in the docker-compose
- you can call another service by its name like `TF_SERVING_HOST=clothing-model:8500` in the above example.

* create and run containers (shot down containers) using docker compose

```commandline
docker-compose -f fileName.yml up
docker-compose -f fileName.yml down
```

`-f` is for file name

---

### difference between **dockerfile** and **docker-compose**

- dockerfile is being used when we want to create a new image.
- when you want to use an existing image (from docker hub) you don't need to create dockerfile
- docker-compose is being used when you want to combine multiple containers and their config into a single file
- docker-compose holds all the configs you may need when running `docker run`
- docker-compose can consist of multiple containers. in docker desktop, ![Untitled](../images/containerIcon.png) is the
  icon of container and ![Untitled](../images/dockerComposeIcon.png) is the icon of docker-compose
- dockerfile is more about the modification and configs of an image, docker compose is more about communication of all
  containers

## Docker Volumes

use for data persistence (db containers)
we plug the physical system file path into the container file path
![volumes.png](../images/volumes.png)

- building a container with volume

```
docker run -v hostPath:containerPath imageName
```

- specify the volume in docker-compose

```
services:
	service1:
		volumes:
			- name1:containerPath
	service2:
		volumes:
			- hostPath:containerPath

volumes:
	name1
```

there are three types of volumes which are:

1. host volumes `hostPath:containerPath`
2. anonymous volumes `containerPath`
3. named volumes `name:containerPath`

## commands cheat sheet

| command                  | use case          | functionality                                    |
|--------------------------|-------------------|--------------------------------------------------|
| `docker pul`             | 🖼️ images        | get the image from the docker hub                |
| `docker run`             | 🖼️ images        | get and create a container from image            |
| `docker images`          | 🖼️ images        | images list                                      |
| `docker rmi`             | 🖼️ images        | remove an image                                  |
| `docker build`           | 🖼️ images        | create an image from a Dockerfile                |
| `docker start`           | 📦 containers     | start the container                              |
| `docker stop`            | 📦 containers     | stop the container                               |
| `docker restart`         | 📦 containers     | restart the container                            |
| `docker ps`              | 📦 containers     | containers list                                  |
| `docker logs`            | 📦 containers     | see logs of a container                          |
| `docker exec`            | 📦 containers     | go to the container's terminal                   |
| `docker rm`              | 📦 containers     | remove a container                               |
| `docker network ls`      | 🌐 network        | networks list                                    |
| `docker network create`  | 🌐 network        | create a network                                 |
| `docker-compose up`      | 🚢 docker-compose | create a docker-compose from docker-compose file |
| `docker-compose down`    | 🚢 docker-compose | remove the docker-compose                        |
| `docker-compose start`   | 🚢 docker-compose | start the created docker-compose                 |
| `docker-compose restart` | 🚢 docker-compose | stop the existing docker-compose |
| `docker-compose stop`    | 🚢 docker-compose | stop the existing docker-compose |
