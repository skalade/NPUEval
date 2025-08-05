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
	-v $(pwd):/host \
	npueval \
	bash -c "cd /host && python3 scripts/check_xrt_versions.py  && python3 $1"
