#!/bin/bash

# Script to setup the environment (Pull Apptainer image)

# Create containers directory if not exists
mkdir -p containers

echo "Pulling PyTorch container with CUDA support..."
# We use a base PyTorch image and will install scandeval at runtime or build a custom image.
# For efficiency, it's better to build a custom image. Here is a definition file generation.

cat <<EOF > containers/euroeval.def
Bootstrap: docker
From: pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

%post
    pip install --upgrade pip
    pip install scandeval
    pip install accelerate bitsandbytes  # Optimizations often needed

%environment
    export LC_ALL=C
    export LANG=C
EOF

echo "Building Apptainer image from definition (Requires sudo/fakeroot usually, or remote build)"
echo "On Berzelius, you might need to just pull a base image and install user packages, or use 'apptainer build --fakeroot' if allowed."
echo "Attempting build..."

apptainer build --fakeroot containers/euroeval.sif containers/euroeval.def

