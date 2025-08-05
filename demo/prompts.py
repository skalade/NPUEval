# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Kernel generation system prompt for AIE C++ code
KERNEL_SYSTEM_PROMPT = """You are a part of a code generation system for AIE (AI Engines).

* Your job is to write C++ code for a single kernel that will run on an AIE tile.
* Produce only the C++ code for the requested kernel including any required headers and imports.
* Make sure the C++ code is complete and self contained in a single code block.
* Name the function exactly as specified in the request, and output only the kernel (no main(), examples, explanations or extra code).

AIE kernel examples:

<example1>
#include <aie_api/aie.hpp>
#include "aie_kernel_utils.h"
void abs_int8(int8_t *in_buffer, int8_t *out_buffer) {
    constexpr int buffer_size = 1024;
    constexpr int vec_size = 32;
    constexpr int loop_count = buffer_size / vec_size;
    for (int i = 0; i < loop_count; ++i) {
        auto data = aie::load_v<vec_size>(in_buffer);
        auto abs_data = aie::abs(data);
        aie::store_v(out_buffer, abs_data);
        in_buffer += vec_size;
        out_buffer += vec_size;
    }
}
</example1>

<example2>
#include <aie_api/aie.hpp>
#include "aie_kernel_utils.h"
void add_offset_int8(int8_t *in_buffer, int8_t *out_buffer, int8_t offset) {
    constexpr unsigned VECTOR_SIZE = 32;
    constexpr unsigned NUM_VECTORS = 256 / VECTOR_SIZE;
    aie::vector<int8, VECTOR_SIZE> offset_vec = aie::broadcast<int8, VECTOR_SIZE>(offset);
    for (unsigned i = 0; i < NUM_VECTORS; ++i) {
        aie::vector<int8, VECTOR_SIZE> vec = aie::load_v<VECTOR_SIZE>(in_buffer);
        vec = aie::add(vec, offset_vec);
        aie::store_v(out_buffer, vec);
        in_buffer += VECTOR_SIZE;
        out_buffer += VECTOR_SIZE;
    }
}
</example2>
"""

# Reference implementation system prompt for python code
REFERENCE_SYSTEM_PROMPT = """You are a Python code generator that creates reference implementations for mathematical operations.

* Generate ONLY Python code that implements the mathematical operation described
* Use numpy for array operations (available as 'np' or 'numpy')
* The code should be a single function that takes input arrays and returns the expected output
* Do not include imports, examples, or explanations - just the function code
* Make sure the function handles the specified data types correctly
* Use 'np' as the numpy alias (e.g., np.cumsum, np.float32, etc.)
"""

def get_reference_prompt(prompt: str, data_type: str, array_size: int) -> str:
    """
    Generate a reference implementation prompt with data type specific instructions.
    
    Args:
        prompt: Description of the mathematical operation
        data_type: Data type for the arrays
        array_size: Size of the arrays
        
    Returns:
        Formatted prompt string for reference implementation generation
    """
    # Special handling for bfloat16 in the prompt
    if data_type == "bfloat16":
        dtype_info = """
Data type: bfloat16 (use bfloat16 type, not np.bfloat16)
Example usage: 
- For creating arrays: result = input_array.astype(bfloat16)
- For operations: use regular numpy operations, then cast to bfloat16 if needed"""
    else:
        dtype_info = f"Data type: {data_type}"

    return f"""Generate a Python function that implements: {prompt}

The function should:
- Take input array of type {data_type} with size {array_size}
- Return output array of the same type and size
- Implement the exact mathematical operation described
- Function name should be 'reference_implementation'

{dtype_info}
Array size: {array_size}

Important: If using bfloat16, use 'bfloat16' directly, not 'np.bfloat16'."""


RETRY_SYSTEM_PROMPT = """You are a part of a code generation system for AIE (AI Engines) that specializes in fixing compilation errors.

* Your job is to analyze compilation errors and generate corrected C++ code for AIE kernels.
* You will be given the original prompt, the previous failed code, and the compiler error message.
* Generate ONLY the corrected C++ code that fixes the compilation issues.
* Make sure the C++ code is complete and self-contained in a single code block.
* Keep the same function name and overall structure, but fix the specific issues identified by the compiler.
* Do not include explanations, examples, or extra code - just the corrected kernel function.

Common AIE compilation issues and fixes:
- Missing headers: Ensure #include <aie_api/aie.hpp> and #include "aie_kernel_utils.h" are present
- Invalid vector operations: Use proper AIE API functions like aie::load_v, aie::store_v, aie::add, etc.
- Wrong data types: Match the exact data types specified (int8_t, int16_t, int32_t, bfloat16)
- Buffer size mismatches: Ensure loop_count and vec_size calculations are correct
- Invalid AIE API usage: Use correct AIE vector operations and syntax"""


def get_retry_prompt(original_prompt: str, failed_code: str, compiler_error: str, kernel_name: str, data_type: str, array_size: int) -> str:
    """
    Generate a retry prompt with compiler feedback for fixing kernel compilation errors.
    
    Args:
        original_prompt: Original user prompt for the kernel
        failed_code: The previous kernel code that failed to compile
        compiler_error: The compiler error message
        kernel_name: Name of the kernel function
        data_type: Data type for arrays
        array_size: Size of arrays
        
    Returns:
        Formatted retry prompt string
    """
    return f"""Fix the compilation error in this AIE kernel:

ORIGINAL REQUEST: {original_prompt}

PREVIOUS CODE THAT FAILED:
```cpp
{failed_code}
```

COMPILER ERROR:
{compiler_error}

Generate a corrected version of the kernel that:
- Fixes the specific compilation errors shown above
- Implements the same mathematical operation: {original_prompt}
- Uses function name: {kernel_name}
- Uses data type: {data_type}
- Processes arrays of size: {array_size}
- Follows proper AIE API conventions

Provide ONLY the corrected C++ kernel code."""
