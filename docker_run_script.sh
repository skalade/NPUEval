#!/bin/bash
#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Copy running host version info from host to docker env
cp /opt/xilinx/xrt/amdxdna/version.json $(pwd)

docker run -it \
	--device=/dev/accel/accel0:/dev/accel/accel0 \
	--cap-add=NET_ADMIN \
	--ulimit memlock=-1 \
	--device=/dev/kfd \
	--device=/dev/dri \
	--group-add video \
	--group-add render \
	-e OLLAMA_MODELS=/host/ollama_models \
	-v $(pwd):/host \
	npueval \
	bash

# if on phoenix add this to enable iGPU
#-e HSA_OVERRIDE_GFX_VERSION=11.0.0 \
