FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu16.04

# Install requirements
RUN apt-get update && \
    apt-get install -y \
        python \
        sudo \
        curl \
        software-properties-common \
        tree \
        nginx-extras \
        vim \
        gettext-base \
        jq \
        build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN printf "Defaults    env_reset\nDefaults	mail_badpass\nDefaults	secure_path=\"/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/snap/bin\"\nroot    ALL = NOPASSWD : ALL\ndocker  ALL = NOPASSWD : ALL\n%sudo   ALL = NOPASSWD : ALL" > /etc/sudoers
RUN chmod 440 /etc/sudoers
RUN adduser --disabled-password --gecos '' docker
RUN adduser docker sudo
RUN chmod 777 /etc
USER docker

RUN sudo chown -R docker:docker /home/docker
RUN sudo chmod -R 755 /home/docker
# Set up paths
WORKDIR /home/docker
ENV HOME /home/docker
ADD environment.yml $HOME/environment.yml
ADD setup-environment.sh $HOME/setup-environment.sh
RUN . $HOME/setup-environment.sh
RUN sudo chmod -R docker:docker /workspace/mmdetection
RUN cd /workspace/mmdetection && ./compile
RUN cd /workspace/mmdetection && python setup.py install