#!/usr/bin/env python3
"""
NPU Kernel Generation Demo

A demo that generates NPU kernels from prompts, builds xclbins, and verifies results.
Limited to 1-input, 1-output kernels as specified.
"""

import os
import re
import json
import numpy as np
from typing import Optional, Dict, Any, Tuple

import openai
from ml_dtypes import bfloat16

# Import npueval modules
from npueval.iron import build_app
from npueval.tools import aie_compiler, build_single_kernel_app
from npueval.executor import NPUExecutor

class NPUKernelDemo:
    """Demo class for generating NPU kernels from prompts."""
    
    def __init__(self, model: str = "gpt-4o-mini", output_dir: str = "demo_results", api_key: Optional[str] = None):
        """Initialize the demo with specified model and output directory."""
        self.model = model
        self.output_dir = output_dir
        self.temperature = 0.4
        
        # Initialize OpenAI client
        if api_key:
            self.client = openai.OpenAI(api_key=api_key)
        else:
            self.client = openai.OpenAI()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # System prompts
        self.kernel_system_prompt = """You are a part of a code generation system for AIE (AI Engines).

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

        self.reference_system_prompt = """You are a Python code generator that creates reference implementations for mathematical operations.

* Generate ONLY Python code that implements the mathematical operation described
* Use numpy for array operations
* The code should be a single function that takes input arrays and returns the expected output
* Do not include imports, examples, or explanations - just the function code
* Make sure the function handles the specified data types correctly
"""
        
    def extract_codeblock(self, text: str) -> Optional[str]:
        """Extract code from markdown codeblocks."""
        code_blocks = re.findall(r'```(?:[a-zA-Z0-9]+)?\n(.*?)```|```(.*?)```', text, re.DOTALL)
        code_blocks = [block for match in code_blocks for block in match if block]
        return code_blocks[0].strip() if code_blocks else None

    def generate_kernel_from_prompt(self, prompt: str, kernel_name: str) -> Dict[str, Any]:
        """
        Generate a kernel from a text prompt using direct OpenAI API call.
        
        Args:
            prompt: Natural language description of the kernel
            kernel_name: Name for the generated kernel function
            
        Returns:
            Dictionary containing generated code and metadata
        """
        print(f"Generating kernel '{kernel_name}' from prompt...")
        
        # Create full prompt with function name specification
        full_prompt = f"{prompt}\nName the function '{kernel_name}'."
        
        # Generate code using OpenAI API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.kernel_system_prompt},
                {"role": "user", "content": full_prompt}
            ],
            temperature=self.temperature,
            seed=42
        )
        
        generated_text = response.choices[0].message.content
        generated_code = self.extract_codeblock(generated_text)
        
        if not generated_code:
            raise ValueError("Failed to extract code from LLM response")
            
        result = {
            'kernel_name': kernel_name,
            'prompt': prompt,
            'generated_code': generated_code,
            'response_text': generated_text,
            'token_usage': response.usage.model_dump()
        }
        
        return result

    def generate_reference_implementation(self, prompt: str, data_type: str, array_size: int) -> str:
        """
        Generate reference Python implementation using LLM.
        
        Args:
            prompt: Description of the mathematical operation
            data_type: Data type for the arrays
            array_size: Size of the arrays
            
        Returns:
            Python function code as string
        """
        print("Generating reference implementation...")
        
        # Special handling for bfloat16 in the prompt
        if data_type == "bfloat16":
            dtype_info = """
Data type: bfloat16 (use bfloat16 type, not np.bfloat16)
Example usage: 
- For creating arrays: result = input_array.astype(bfloat16)
- For operations: use regular numpy operations, then cast to bfloat16 if needed"""
        else:
            dtype_info = f"Data type: {data_type}"

        reference_prompt = f"""Generate a Python function that implements: {prompt}

The function should:
- Take input array of type {data_type} with size {array_size}
- Return output array of the same type and size
- Implement the exact mathematical operation described
- Function name should be 'reference_implementation'

{dtype_info}
Array size: {array_size}

Important: If using bfloat16, use 'bfloat16' directly, not 'np.bfloat16'."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.reference_system_prompt},
                {"role": "user", "content": reference_prompt}
            ],
            temperature=0.1,  # Lower temperature for more deterministic reference
            seed=42
        )
        
        generated_text = response.choices[0].message.content
        reference_code = self.extract_codeblock(generated_text)
        
        if not reference_code:
            # If no code block found, assume the entire response is code
            reference_code = generated_text.strip()
            
        return reference_code
    
    def create_test_arrays(self, prompt: str, data_type: str = "int8", size: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create input and reference output arrays using LLM-generated reference implementation.
        
        Args:
            prompt: Description of the mathematical operation for reference generation
            data_type: Data type for arrays ("int8", "int16", "int32", "bfloat16")
            size: Size of the arrays
            
        Returns:
            Tuple of (input_array, reference_output_array)
        """
        np.random.seed(42)  # For reproducible results
        
        # Generate input array
        if data_type == "int8":
            input_array = np.random.randint(-128, 127, size=size, dtype=np.int8)
        elif data_type == "int16":
            input_array = np.random.randint(-32768, 32767, size=size, dtype=np.int16)
        elif data_type == "int32":
            input_array = np.random.randint(-2147483648, 2147483647, size=size, dtype=np.int32)
        elif data_type == "bfloat16":
            input_array = np.random.randn(size).astype(bfloat16)
        else:
            raise ValueError(f"Unsupported data type: {data_type}. NPU supports only int8, int16, int32, and bfloat16.")
        
        # Generate reference implementation using LLM
        reference_code = self.generate_reference_implementation(prompt, data_type, size)
        
        # Execute the reference code to get expected output
        local_vars = {'np': np, 'input_array': input_array}
        global_vars = {'np': np, 'bfloat16': bfloat16}
        
        try:
            # Execute the generated reference function
            exec(reference_code, global_vars, local_vars)
            
            # Call the reference implementation
            if 'reference_implementation' in local_vars:
                reference_output = local_vars['reference_implementation'](input_array)
            else:
                raise ValueError("Generated reference code doesn't contain 'reference_implementation' function")
                
            # Ensure output has correct dtype
            if data_type == "int8":
                reference_output = reference_output.astype(np.int8)
            elif data_type == "int16":
                reference_output = reference_output.astype(np.int16)
            elif data_type == "int32":
                reference_output = reference_output.astype(np.int32)
            elif data_type == "bfloat16":
                reference_output = reference_output.astype(bfloat16)
            
        except Exception as e:
            print(f"Error executing reference implementation: {e}")
            print(f"Generated reference code:\n{reference_code}")
            raise ValueError(f"Failed to execute LLM-generated reference implementation: {e}")
            
        return input_array, reference_output
    
    def build_xclbin(self, kernel_code: str, kernel_name: str, 
                    input_array: np.ndarray, output_array: np.ndarray) -> Dict[str, Any]:
        """
        Build xclbin from kernel code and test arrays.
        
        Args:
            kernel_code: Generated C++ kernel code
            kernel_name: Name of the kernel function
            input_array: Input test array
            output_array: Expected output array
            
        Returns:
            Dictionary with build results and file paths
        """
        print(f"Building xclbin for kernel '{kernel_name}'...")
        
        # Add wrapper code that npueval expects
        wrapper_name = f"{kernel_name}_wrapper"
        full_kernel_code = kernel_code + f"""

extern "C" {{
    void {wrapper_name}(int8_t *in_buffer, int8_t *out_buffer) {{
        ::aie::set_rounding(aie::rounding_mode::positive_inf);
        event0();
        {kernel_name}(in_buffer, out_buffer);
        event1();
    }}
}}"""
        
        # Compile kernel with wrapper
        compile_result = aie_compiler(
            full_kernel_code,
            kernel_name=wrapper_name,
            output_dir=self.output_dir,
            compiler="peano",
            dev=os.environ.get('NPU', 'npu1_1col'),
            verbose_output=False
        )
        
        if not compile_result.startswith('Compilation successful.'):
            raise RuntimeError(f"Kernel compilation failed: {compile_result}")
        
        # Generate MLIR using numpy arrays directly
        tile_size = input_array.size
        trace_size = 8192
        
        mlir, padding = build_app(
            wrapper_name,
            [input_array],  # Pass numpy array directly
            output_array,   # Pass numpy array directly
            [],  # No RTPs for simple 1-in-1-out kernels
            tile_size=tile_size,
            trace_size=trace_size,
            dev=os.environ.get('NPU', 'npu1_1col')
        )
        
        if not mlir:
            raise RuntimeError("Failed to generate MLIR")
            
        # Save MLIR file
        mlir_path = f"{self.output_dir}/{wrapper_name}.mlir"
        with open(mlir_path, 'w') as f:
            f.write(mlir)
        
        # Build application
        build_result = build_single_kernel_app(
            mlir_path,
            f"{self.output_dir}/{wrapper_name}.o",
            output_dir=self.output_dir,
            xclbin_name=wrapper_name,
            compiler_backend="peano"
        )
        
        if build_result.returncode != 0:
            raise RuntimeError(f"Application build failed with return code {build_result.returncode}")
        
        result = {
            'xclbin_path': f"{self.output_dir}/{wrapper_name}.xclbin",
            'instr_path': f"{self.output_dir}/{wrapper_name}.bin",
            'mlir_path': mlir_path,
            'padding': padding,
            'compile_result': compile_result
        }
        
        return result
    
    def verify_kernel(self, xclbin_path: str, instr_path: str, 
                     input_array: np.ndarray, expected_output: np.ndarray,
                     padding: int = 0) -> Dict[str, Any]:
        """
        Verify kernel execution on NPU against expected output.
        
        Args:
            xclbin_path: Path to compiled xclbin file
            instr_path: Path to instruction file
            input_array: Input test data
            expected_output: Expected output for validation
            padding: Padding value from MLIR generation
            
        Returns:
            Dictionary with verification results
        """
        print("Running kernel verification on NPU...")
        
        # Create executor
        executor = NPUExecutor(
            xclbin=xclbin_path,
            instr=instr_path,
            verbose=True
        )
        
        # Run kernel
        trace_name = f"{self.output_dir}/verification_trace.txt"
        results = executor.run(
            in_buffers=[input_array],
            out_buffers=[expected_output],
            trace_size=8192,
            trace_name=trace_name,
            padding=padding
        )
        
        if isinstance(results, tuple):
            eval_output, total_cycles, vector_cycles = results
            verification_result = {
                'success': eval_output['success'],
                'stats': eval_output['stats'],
                'total_cycles': total_cycles,
                'vector_cycles': vector_cycles,
                'vector_score': vector_cycles/total_cycles if total_cycles > 0 else 0,
                'trace_file': trace_name
            }
        else:
            verification_result = {
                'success': results['success'],
                'stats': results['stats'],
                'trace_file': trace_name
            }
        
        return verification_result
    
    def run_demo(self, prompt: str, kernel_name: str, 
                data_type: str = "int8", array_size: int = 1024) -> Dict[str, Any]:
        """
        Run the complete demo pipeline: generate -> build -> verify.
        
        Args:
            prompt: Natural language kernel description
            kernel_name: Name for the kernel function
            data_type: Data type for test arrays
            array_size: Size of test arrays
            
        Returns:
            Dictionary with complete demo results
        """
        print(f"\n=== NPU Kernel Generation Demo ===")
        print(f"Prompt: {prompt}")
        print(f"Kernel: {kernel_name}")
        print(f"Data type: {data_type}, Array size: {array_size}")
        print("=" * 50)
        
        generation_result = None
        input_array = None
        expected_output = None
        build_result = None
        verification_result = None
        
        try:
            # Step 1: Generate kernel code
            generation_result = self.generate_kernel_from_prompt(prompt, kernel_name)
            
            # Step 2: Create test arrays using LLM-generated reference
            input_array, expected_output = self.create_test_arrays(prompt, data_type, array_size)
            
            # Step 3: Build xclbin
            build_result = self.build_xclbin(
                generation_result['generated_code'],
                kernel_name,
                input_array,
                expected_output
            )
            
            # Step 4: Verify on NPU
            verification_result = self.verify_kernel(
                build_result['xclbin_path'],
                build_result['instr_path'],
                input_array,
                expected_output,
                build_result['padding']
            )
            
            # Compile complete results
            demo_result = {
                'success': verification_result['success'],
                'generation': generation_result,
                'build': build_result,
                'verification': verification_result,
                'test_data': {
                    'input_shape': input_array.shape,
                    'input_dtype': str(input_array.dtype),
                    'output_shape': expected_output.shape,
                    'output_dtype': str(expected_output.dtype)
                }
            }
            
            # Save results
            results_file = f"{self.output_dir}/{kernel_name}_demo_results.json"
            with open(results_file, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                json_result = demo_result.copy()
                json_result['test_data']['input_sample'] = input_array[:10].tolist()
                json_result['test_data']['expected_output_sample'] = expected_output[:10].tolist()
                json.dump(json_result, f, indent=2)
            
            print(f"\n=== Demo Results ===")
            print(f"Success: {demo_result['success']}")
            if verification_result.get('stats'):
                print(f"Accuracy: {verification_result['stats']}")
            if verification_result.get('vector_score'):
                print(f"Vectorization Score: {verification_result['vector_score']:.3f}")
            print(f"Results saved to: {results_file}")
            
            return demo_result
            
        except Exception as e:
            # Build error result, preserving any successful steps
            error_result = {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__
            }
            
            # Preserve generation result if it was successful
            if generation_result:
                error_result['generation'] = generation_result
            
            # Preserve test data if arrays were created
            if input_array is not None and expected_output is not None:
                error_result['test_data'] = {
                    'input_shape': input_array.shape,
                    'input_dtype': str(input_array.dtype),
                    'output_shape': expected_output.shape,
                    'output_dtype': str(expected_output.dtype)
                }
            
            # Preserve build result if it was successful
            if build_result:
                error_result['build'] = build_result
                
            # Preserve verification result if it was attempted
            if verification_result:
                error_result['verification'] = verification_result
            
            error_file = f"{self.output_dir}/{kernel_name}_error.json"
            with open(error_file, 'w') as f:
                # Handle numpy arrays for JSON serialization
                json_result = error_result.copy()
                if input_array is not None and expected_output is not None:
                    json_result['test_data']['input_sample'] = input_array[:10].tolist()
                    json_result['test_data']['expected_output_sample'] = expected_output[:10].tolist()
                json.dump(json_result, f, indent=2)
                
            print(f"\n=== Demo Failed ===")
            print(f"Error: {e}")
            print(f"Error details saved to: {error_file}")
            
            return error_result

def main():
    """Run demo with example ReLU kernel."""
    # Check NPU environment
    if 'NPU' not in os.environ:
        os.environ['NPU'] = 'npu1_1col'  # Default NPU device
    
    demo = NPUKernelDemo(model="gpt-4o-mini", output_dir="demo_results")
    
    # Example: Generate ReLU kernel
    prompt = "Write a ReLU kernel that takes in vectors of 1024 elements of int8."
    kernel_name = "relu_int8_demo"
    
    result = demo.run_demo(prompt, kernel_name, data_type="int8", array_size=1024)
    
    return result

if __name__ == "__main__":
    main()
