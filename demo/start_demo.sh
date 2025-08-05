#!/bin/bash
#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

docker run -it \
	--device=/dev/accel/accel0:/dev/accel/accel0 \
	--cap-add=NET_ADMIN \
	--ulimit memlock=-1 \
	--device=/dev/kfd \
	--device=/dev/dri \
	--group-add video \
	--group-add render \
	-p 8501:8501 \
	-e OLLAMA_MODELS=/host/ollama_models \
	-e OPENAI_API_KEY=$OPENAI_API_KEY \
	-e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
	-v $(pwd):/host \
	npueval \
	bash -c "cd /host && streamlit run kernel_dashboard.py"

# if on phoenix add this to enable iGPU
#-e HSA_OVERRIDE_GFX_VERSION=11.0.0 \
