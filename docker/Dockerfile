FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    curl \
    git \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    software-properties-common \
    net-tools \
    vim \
    virtualenv \
    wget \
    xpra \
    xserver-xorg-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN DEBIAN_FRONTEND=noninteractive add-apt-repository --yes ppa:deadsnakes/ppa && apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install --yes python3.8-dev python3.8 python3-pip
RUN virtualenv --python=python3.8 env


RUN rm /usr/bin/python
RUN ln -s /env/bin/python3.8 /usr/bin/python
RUN ln -s /env/bin/pip3.8 /usr/bin/pip
RUN ln -s /env/bin/pytest /usr/bin/pytest

RUN curl -o /usr/local/bin/patchelf https://s3-us-west-2.amazonaws.com/openai-sci-artifacts/manual-builds/patchelf_0.9_amd64.elf \
    && chmod +x /usr/local/bin/patchelf

ENV LANG C.UTF-8

RUN mkdir -p /root/.mujoco \
    && wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz \
    && tar -xf mujoco.tar.gz -C /root/.mujoco \
    && rm mujoco.tar.gz

ENV LD_LIBRARY_PATH /root/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:${LD_LIBRARY_PATH}


# Workaround for https://bugs.launchpad.net/ubuntu/+source/nvidia-graphics-drivers-375/+bug/1674677


WORKDIR /mujoco_py
# Copy over just requirements.txt at first. That way, the Docker cache doesn't
# expire until we actually change the requirements.
COPY ./requirements.txt /mujoco_py/

RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir /install
WORKDIR /install
RUN pip install setuptools

RUN pip install gym==0.10.8
RUN pip install mujoco_py
#RUN git clone https://github.com/oxwhirl/sacred.git /install/sacred && cd /install/sacred && python3 setup.py install
RUN pip install git+https://github.com/IDSIA/sacred.git

WORKDIR /mujoco_py
# RUN pip install git+https://github.com/schroederdewitt/multiagent_mujoco.git
# Delay moving in the entire code until the very end.
#ENTRYPOINT ["/mujoco_py/vendor/Xdummy-entrypoint"]
# CMD ["pytest"]
COPY . /mujoco_py
COPY /assets /env/lib/python3.8/site-packages/multiagent_mujoco/assets
