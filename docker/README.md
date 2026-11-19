To ease the process of installing all the dependencies, we provide a Dockerfile and a simple guideline to build a Docker image with all of above installed. The Docker image is built on top of a platform-specific base image, and it contains all the dependencies required to run the experiments. We provide Dockerfiles for NVIDIA GPUs, AMD GPUs, and Ascend NPUs.

```bash
git clone --recursive https://github.com/tile-ai/tilelang TileLang
cd TileLang/docker
# build the image, this may take a while (around 10+ minutes on our test machine)
# replace the version number cu124 with the one you want to use
# replace .cu** with .rocm for AMD GPU
docker build -t tilelang_workspace -f Dockerfile.cu124 .
# run the container
# if it's nvidia
docker run -it --cap-add=SYS_ADMIN --network=host --gpus all --cap-add=SYS_PTRACE --shm-size=4G --security-opt seccomp=unconfined --security-opt apparmor=unconfined --name tilelang_test tilelang_workspace bash
# if it's amd
docker run -it --cap-add=SYS_ADMIN --network=host --device=/dev/kfd --device=/dev/dri  --cap-add=SYS_PTRACE --shm-size=4G --security-opt seccomp=unconfined --security-opt apparmor=unconfined --name tilelang_test tilelang_workspace bash
```

## Ascend 950

From the `docker/` directory, build the Linux amd64 image for TileLang v0.1.15
and CANN 9.2.0 on Ubuntu 24.04:

```bash
docker build --platform linux/amd64 -t tilelang_workspace -f Dockerfile.ascend .
```

On a host with a working Ascend driver and Ascend Docker Runtime, set `NPU_ID`
to the device allocated to you and create a development container:

```bash
NPU_ID=0
docker run -it --runtime=ascend \
  -e "ASCEND_VISIBLE_DEVICES=${NPU_ID}" \
  --name tilelang_test tilelang_workspace bash
```
