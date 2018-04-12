#!/bin/bash

set -ex

# Install common dependencies
apt-get update
apt-get install -y --no-install-recommends python3-dev python3-pip python3-setuptools

pip3 install --upgrade pip
pip3 install numpy decorator six future setuptools pyyaml
# install the devel needed for aten
pip3 install mkl-devel typing

which python3
python3 --version
python3 -c 'import yaml'

# Cleanup package manager
apt-get autoclean && apt-get clean
rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
