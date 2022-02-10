#!/usr/bin/env bash

echo "Installing Docker daemon with NVIDIA runtime..."

# Docker
curl https://get.docker.com | sh &&
sudo systemctl --now enable docker

# Nvidia Docker
distribution=$(
    . /etc/os-release
    echo $ID$VERSION_ID
) &&
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - &&
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Add docker to sudo group
echo "Adding docker to sudo group..."
sudo groupadd docker
sudo usermod -aG docker "${USER}"
