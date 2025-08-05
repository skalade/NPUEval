# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Preset kernel configurations for the dashboard
PRESET_CONFIGS = {
    "Custom (enter your own)": {
        "prompt": "",
        "kernel_name": "my_kernel",
        "data_type": "int8",
        "array_size": 1024
    },
    "ReLU Activation": {
        "prompt": "Write a ReLU kernel that takes in vectors of 1024 int8 elements and applies ReLU activation (max(0, x)) to each element.",
        "kernel_name": "relu_kernel",
        "data_type": "int8",
        "array_size": 1024
    },
    "ReLU Activation (Scalar)": {
        "prompt": "Write a ReLU kernel that takes in a vector of 1024 int8 elements and applies ReLU activation (max(0, x)) to each element. Do not use AIE APIs.",
        "kernel_name": "relu_kernel",
        "data_type": "int8",
        "array_size": 1024
    },
    "Add Offset": {
        "prompt": "Write a kernel that adds a constant offset of 5 to the input array of 1024 elements with dtype of int16.",
        "kernel_name": "add_offset",
        "data_type": "int16",
        "array_size": 1024
    },
    "Cumsum": {
        "prompt": "This AIE kernel computes the elementwise cumulative sum (cumsum) of a bfloat16 input vector of length 256. Each output element is the sum of all input elements up to and including that position.",
        "kernel_name": "cumsum",
        "data_type": "bfloat16",
        "array_size": 256
    },
    "Negate": {
        "prompt": "Negate each element in a vector of int8_t (length 512). The input is a buffer of 512 int8, and the output buffer receives the elementwise negation.",
        "kernel_name": "negate",
        "data_type": "int8",
        "array_size": 512
    },
    "Sigmoid Activation": {
        "prompt": "Write a sigmoid activation kernel that applies the sigmoid function (1/(1+exp(-x))) to each element of an input array of 256 bfloat16 values.",
        "kernel_name": "sigmoid_kernel",
        "data_type": "bfloat16",
        "array_size": 256
    },
    "Softmax": {
        "prompt": "Write a softmax kernel that computes the softmax function on an input vector of 512 bfloat16 elements.",
        "kernel_name": "softmax_kernel",
        "data_type": "bfloat16",
        "array_size": 512
    }
}

# NPU-compatible data types
SUPPORTED_DATA_TYPES = ["int8", "int16", "int32", "bfloat16"]
