# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

from setuptools import find_packages, setup

setup(
    name="npueval",
    version='0.1',
    packages=find_packages(),
    package_data={
    '': ['dataset/npueval.jsonl'],
    },
    install_requires=[
        "numpy",
        "openai",
        "anthropic",
        "CppHeaderParser",
        "ml_dtypes",
        "llama_index",
        "pandas",
        "seaborn",
        "streamlit",
        "plotly"
    ],
    description="Utils for NPUEval dataset generation and evaluation.")
