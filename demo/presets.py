#!/usr/bin/env python3
"""
Preset configurations for the NPU Kernel Dashboard.

This module contains predefined kernel configurations that users can select
from in the dashboard interface.
"""

# Preset kernel configurations for the dashboard
PRESET_CONFIGS = {
    "Custom (enter your own)": {
        "prompt": "",
        "kernel_name": "my_kernel",
        "data_type": "int8",
        "array_size": 1024
    },
    "ReLU Activation": {
        "prompt": "Write a ReLU kernel that takes in vectors of elements and applies ReLU activation (max(0, x)) to each element",
        "kernel_name": "relu_kernel",
        "data_type": "int8",
        "array_size": 1024
    },
    "Add Offset": {
        "prompt": "Write a kernel that adds a constant offset of 5 to the input array.",
        "kernel_name": "add_offset",
        "data_type": "int16",
        "array_size": 1024
    },
    "Argmax": {
        "prompt": "Return the index of the largest element of the input array.",
        "kernel_name": "argmax",
        "data_type": "int32",
        "array_size": 256
    },
    "Negate": {
        "prompt": "Negate each element in a vector of int8_t (length 512). The input is a buffer of 512 int8, and the output buffer receives the elementwise negation.",
        "kernel_name": "negate",
        "data_type": "int8",
        "array_size": 512
    },
    "Sigmoid Activation": {
        "prompt": "Write a sigmoid activation kernel that applies the sigmoid function (1/(1+exp(-x))) to each element",
        "kernel_name": "sigmoid_kernel",
        "data_type": "bfloat16",
        "array_size": 1024
    },
    "Softmax": {
        "prompt": "Write a softmax kernel that computes the softmax function across the last dimension of the input",
        "kernel_name": "softmax_kernel",
        "data_type": "bfloat16",
        "array_size": 1024
    }
}

# NPU-compatible data types
SUPPORTED_DATA_TYPES = ["int8", "int16", "int32", "bfloat16"]