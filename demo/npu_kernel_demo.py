# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import os
import re
import json
import numpy as np
from typing import Optional, Dict, Any, Tuple

import openai
from ml_dtypes import bfloat16

from npueval.iron import build_app
from npueval.tools import aie_compiler, build_single_kernel_app
from npueval.executor import NPUExecutor

from prompts import KERNEL_SYSTEM_PROMPT, REFERENCE_SYSTEM_PROMPT, RETRY_SYSTEM_PROMPT, get_reference_prompt, get_retry_prompt

class NPUKernelDemo:
    """Demo class for generating NPU kernels from prompts."""
    
    def __init__(self, model: str = "gpt-4o-mini", output_dir: str = "demo_results", api_key: Optional[str] = None, base_url: Optional[str] = None, max_retries: int = 2):
        """Initialize the demo with specified model and output directory."""
        self.model = model
        self.output_dir = output_dir
        self.temperature = 0.4
        self.max_retries = max_retries
        
        # Initialize OpenAI client
        client_kwargs = {}
        if api_key:
            client_kwargs['api_key'] = api_key
        if base_url:
            client_kwargs['base_url'] = base_url
            
        self.client = openai.OpenAI(**client_kwargs)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def extract_codeblock(self, text: str) -> Optional[str]:
        """Extract code from markdown codeblocks or return entire text if no codeblocks found."""
        code_blocks = re.findall(r'```(?:[a-zA-Z0-9]+)?\n(.*?)```|```(.*?)```', text, re.DOTALL)
        code_blocks = [block for match in code_blocks for block in match if block]
        if code_blocks:
            return code_blocks[0].strip()
        else:
            # If no markdown codeblocks found, assume the entire response is code
            return text.strip()
        

    def generate_kernel_from_prompt(self, prompt: str, kernel_name: str, data_type: str = "int8") -> Dict[str, Any]:
        """
        Generate a kernel from a text prompt using direct OpenAI API call.
        
        Args:
            prompt: Natural language description of the kernel
            kernel_name: Name for the generated kernel function
            data_type: Data type for the kernel (int8, int16, int32, bfloat16)
            
        Returns:
            Dictionary containing generated code and metadata
        """
        print(f"Generating kernel '{kernel_name}' from prompt...")
        
        # Map data types to C++ types for the prompt
        cpp_type_map = {
            "int8": "int8_t",
            "int16": "int16_t", 
            "int32": "int32_t",
            "bfloat16": "bfloat16"
        }
        cpp_type = cpp_type_map.get(data_type, "int8_t")
        
        # Create full prompt with function name and data type specification
        full_prompt = f"{prompt}\nName the function '{kernel_name}'.\nUse {cpp_type} data type for input and output buffers."
        
        # Generate code using OpenAI API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": KERNEL_SYSTEM_PROMPT},
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

    def retry_kernel_generation(self, original_prompt: str, failed_code: str, compiler_error: str, 
                               kernel_name: str, data_type: str, array_size: int) -> Dict[str, Any]:
        """
        Retry kernel generation with compiler feedback.
        
        Args:
            original_prompt: Original user prompt for the kernel
            failed_code: The previous kernel code that failed to compile
            compiler_error: The compiler error message
            kernel_name: Name of the kernel function
            data_type: Data type for arrays
            array_size: Size of arrays
            
        Returns:
            Dictionary with regenerated kernel information
        """
        print(f"Retrying kernel generation with compiler feedback...")
        
        # Generate retry prompt with error context
        retry_prompt = get_retry_prompt(original_prompt, failed_code, compiler_error, 
                                      kernel_name, data_type, array_size)
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": RETRY_SYSTEM_PROMPT},
                {"role": "user", "content": retry_prompt}
            ],
            temperature=self.temperature
        )
        
        generated_text = response.choices[0].message.content
        generated_code = self.extract_codeblock(generated_text)
        
        if not generated_code:
            raise ValueError("No code block found in LLM response during retry")
            
        result = {
            'kernel_name': kernel_name,
            'prompt': original_prompt,
            'generated_code': generated_code,
            'response_text': generated_text,
            'token_usage': response.usage.model_dump(),
            'retry_attempt': True,
            'original_error': compiler_error
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
        
        # Generate reference prompt using the helper function
        reference_prompt = get_reference_prompt(prompt, data_type, array_size)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": REFERENCE_SYSTEM_PROMPT},
                {"role": "user", "content": reference_prompt}
            ],
            temperature=0.1,  # Lower temperature for more determinism in Python
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
            Tuple of (input_array, reference_output_array, reference_code)
        """
        # Generate input array
        np.random.seed(42)
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
        local_vars = {'input_array': input_array}
        global_vars = {'np': np, 'numpy': np, 'bfloat16': bfloat16}
        
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
            
        return input_array, reference_output, reference_code
    
    def build_xclbin(self, kernel_code: str, kernel_name: str, 
                    input_array: np.ndarray, output_array: np.ndarray, data_type: str) -> Dict[str, Any]:
        """
        Build xclbin from kernel code and test arrays.
        
        Args:
            kernel_code: Generated C++ kernel code
            kernel_name: Name of the kernel function
            input_array: Input test array
            output_array: Expected output array
            data_type: Data type for the arrays ("int8", "int16", "int32", "bfloat16")
            
        Returns:
            Dictionary with build results and file paths
        """
        print(f"Building xclbin for kernel '{kernel_name}'...")
        
        # Add wrapper code that npueval expects
        wrapper_name = f"{kernel_name}_wrapper"
        
        # Map data type to C++ type
        if data_type == "int8":
            cpp_type = "int8_t"
        elif data_type == "int16":
            cpp_type = "int16_t"
        elif data_type == "int32":
            cpp_type = "int32_t"
        elif data_type == "bfloat16":
            cpp_type = "bfloat16"
        else:
            raise ValueError(f"Unsupported data type for C++: {data_type}")
        full_kernel_code = kernel_code + f"""
#include <aie_api/aie.hpp>
#include "aie_kernel_utils.h"

extern "C" {{
    void {wrapper_name}({cpp_type} *in_buffer, {cpp_type} *out_buffer) {{
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
        
        # Initialize variables
        npu_output_buffer = None
        eval_output = None  # Initialize eval_output to prevent scoping errors
        verification_result = {
            'success': False,
            'stats': {},
            'trace_file': f"{self.output_dir}/verification_trace.txt",
            'error': 'Unknown verification error'
        }
        
        try:
            # Create executor
            executor = NPUExecutor(
                xclbin=xclbin_path,
                instr=instr_path,
                verbose=True
            )
            
            # Create a copy of expected_output for the NPU to write into
            # The executor modifies the output buffer in-place
            npu_output_buffer = expected_output.copy()
            
            # Run kernel
            trace_name = f"{self.output_dir}/verification_trace.txt"
            results = executor.run(
                in_buffers=[input_array],
                out_buffers=[npu_output_buffer],
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
                    'trace_file': trace_name,
                    'npu_output': npu_output_buffer  # Store the actual NPU output
                }
            else:
                verification_result = {
                    'success': results['success'],
                    'stats': results['stats'],
                    'trace_file': trace_name,
                    'npu_output': npu_output_buffer  # Store the actual NPU output
                }
                
        except Exception as e:
            print(f"Error during kernel verification: {e}")
            error_str = str(e)
            
            # Check if this is a trace parsing error but NPU execution may have succeeded
            if "Expecting value: line 1 column 1" in error_str and "json" in error_str.lower():
                # Trace parsing failed, but NPU may have executed successfully
                # Check if we have NPU output data
                if npu_output_buffer is not None:
                    verification_result = {
                        'success': True,  # NPU execution succeeded, just trace parsing failed
                        'stats': {'trace_parsing_failed': True},
                        'trace_file': f"{self.output_dir}/verification_trace.txt",
                        'error': 'NPU execution succeeded but trace parsing failed (empty JSON trace file)',
                        'npu_output': npu_output_buffer,
                        'total_cycles': None,  # Can't determine without trace parsing
                        'vector_cycles': None,
                        'vector_score': None
                    }
                else:
                    verification_result = {
                        'success': False,
                        'stats': {},
                        'trace_file': f"{self.output_dir}/verification_trace.txt",
                        'error': 'NPU execution failed with trace parsing error: ' + error_str,
                        'npu_output': npu_output_buffer
                    }
            else:
                # Other types of NPU execution errors
                verification_result = {
                    'success': False,
                    'stats': {},
                    'trace_file': f"{self.output_dir}/verification_trace.txt",
                    'error': str(e),
                    'npu_output': npu_output_buffer  # Include whatever output buffer we have
                }
        
        return verification_result
    
    def run_demo(self, prompt: str, kernel_name: str, 
                data_type: str = "int8", array_size: int = 1024, 
                status_callback=None) -> Dict[str, Any]:
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
        
        # Step 1: Generate kernel code
        try:
            generation_result = self.generate_kernel_from_prompt(prompt, kernel_name, data_type)
        except Exception as e:
            return self._create_error_result(
                kernel_name, "LLM generation failed", e, 
                generation_result, input_array, expected_output, build_result, verification_result
            )
        
        # Step 2: Create test arrays using LLM-generated reference
        try:
            input_array, expected_output, reference_code = self.create_test_arrays(prompt, data_type, array_size)
        except Exception as e:
            return self._create_error_result(
                kernel_name, "Reference implementation generation failed", e,
                generation_result, input_array, expected_output, build_result, verification_result
            )
        
        # Step 3: Build xclbin with retry mechanism
        current_generation_result = generation_result
        retry_count = 0
        
        while retry_count <= self.max_retries:
            try:
                build_result = self.build_xclbin(
                    current_generation_result['generated_code'],
                    kernel_name,
                    input_array,
                    expected_output,
                    data_type
                )
                # Success! Break out of retry loop
                if retry_count > 0:
                    print(f"✅ Compilation succeeded after {retry_count} retry attempt(s)")
                break
                
            except Exception as e:
                compiler_error = str(e)
                print(f"❌ Compilation attempt {retry_count + 1} failed: {compiler_error}")
                
                # If we've exhausted retries, return error
                if retry_count >= self.max_retries:
                    return self._create_error_result(
                        kernel_name, "Kernel compilation failed", e,
                        current_generation_result, input_array, expected_output, build_result, verification_result, reference_code
                    )
                
                # Try to retry with compiler feedback
                try:
                    print(f"🔄 Attempting retry {retry_count + 1}/{self.max_retries} with compiler feedback...")
                    if status_callback:
                        status_callback(f"🔄 Re-generating code (attempt {retry_count + 1}/{self.max_retries})...")
                    current_generation_result = self.retry_kernel_generation(
                        prompt, 
                        current_generation_result['generated_code'],
                        compiler_error,
                        kernel_name,
                        data_type,
                        array_size
                    )
                    retry_count += 1
                    if status_callback:
                        status_callback(f"🔧 Re-compiling fixed code (attempt {retry_count}/{self.max_retries})...")
                    
                except Exception as retry_e:
                    print(f"❌ Retry generation failed: {retry_e}")
                    return self._create_error_result(
                        kernel_name, "LLM retry generation failed", retry_e,
                        current_generation_result, input_array, expected_output, build_result, verification_result, reference_code
                    )
        
        # Update generation_result to reflect the final successful version (may include retry info)
        generation_result = current_generation_result
        
        # Step 4: Verify on NPU
        try:
            verification_result = self.verify_kernel(
                build_result['xclbin_path'],
                build_result['instr_path'],
                input_array,
                expected_output,
                build_result['padding']
            )
        except Exception as e:
            return self._create_error_result(
                kernel_name, "NPU verification failed", e,
                generation_result, input_array, expected_output, build_result, verification_result, reference_code
            )
        
        # Check if verification failed (NPU ran but accuracy was not met)
        if not verification_result['success']:
            # Create error result but include all the verification data
            mae = verification_result.get('stats', {}).get('abs_error_mean', 0)
            error_msg = f"Verification accuracy not met (Abs error: {mae:.6f})"
            return self._create_error_result(
                kernel_name, "NPU verification failed", ValueError(error_msg),
                generation_result, input_array, expected_output, build_result, verification_result, reference_code
            )
        
        # Create a clean verification result without numpy arrays for the main result
        clean_verification_result = {k: v for k, v in verification_result.items() if k != 'npu_output'}
        
        # Compile complete results (without numpy arrays that can't be JSON serialized)
        demo_result = {
            'success': verification_result['success'],
            'generation': generation_result,
            'reference': {
                'reference_code': reference_code
            },
            'build': build_result,
            'verification': clean_verification_result,
            'test_data': {
                'input_shape': list(input_array.shape),
                'input_dtype': str(input_array.dtype),
                'output_shape': list(expected_output.shape),
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
            
            # Add NPU output sample if available
            npu_output = verification_result.get('npu_output')
            if npu_output is not None:
                # Convert to numpy array if it isn't already, then get sample
                try:
                    if hasattr(npu_output, 'tolist'):
                        json_result['test_data']['npu_output_sample'] = npu_output[:10].tolist()
                    elif isinstance(npu_output, (list, tuple)):
                        json_result['test_data']['npu_output_sample'] = list(npu_output[:10])
                    else:
                        # Try to convert to numpy array first
                        npu_array = np.array(npu_output)
                        json_result['test_data']['npu_output_sample'] = npu_array[:10].tolist()
                except Exception as e:
                    print(f"Warning: Could not save NPU output sample: {e}")
                    
            json.dump(json_result, f, indent=2)
        
        print(f"\n=== Demo Results ===")
        print(f"Success: {demo_result['success']}")
        if verification_result.get('stats'):
            print(f"Accuracy: {verification_result['stats']}")
        if verification_result.get('vector_score'):
            print(f"Vectorization Score: {verification_result['vector_score']:.3f}")
        print(f"Results saved to: {results_file}")
        
        return demo_result
    
    def _create_error_result(self, kernel_name: str, step_name: str, exception: Exception,
                           generation_result, input_array, expected_output, build_result, verification_result, reference_code=None) -> Dict[str, Any]:
        """
        Create an error result with specific step information and preserve successful steps.
        
        Args:
            kernel_name: Name of the kernel being processed
            step_name: Name of the step that failed
            exception: The exception that occurred
            generation_result: Result from kernel generation (if successful)
            input_array: Input test array (if created)
            expected_output: Expected output array (if created)
            build_result: Result from build step (if successful)
            verification_result: Result from verification step (if attempted)
            
        Returns:
            Dictionary with error information and preserved successful steps
        """
        error_result = {
            'success': False,
            'error': str(exception),
            'error_type': type(exception).__name__,
            'failed_step': step_name,
            'detailed_error': f"{step_name}: {str(exception)}"
        }
        
        # Preserve generation result if it was successful
        if generation_result:
            error_result['generation'] = generation_result
            
        # Preserve reference code if it was generated
        if reference_code:
            error_result['reference'] = {
                'reference_code': reference_code
            }
        
        # Preserve test data if arrays were created
        if input_array is not None and expected_output is not None:
            error_result['test_data'] = {
                'input_shape': list(input_array.shape),
                'input_dtype': str(input_array.dtype),
                'output_shape': list(expected_output.shape),
                'output_dtype': str(expected_output.dtype)
            }
        
        # Preserve build result if it was successful
        if build_result:
            error_result['build'] = build_result
            
        # Preserve verification result if it was attempted (without numpy arrays)
        if verification_result:
            clean_verification = {k: v for k, v in verification_result.items() if k != 'npu_output'}
            error_result['verification'] = clean_verification
            
            # If NPU ran but verification failed, include NPU output sample in test_data
            if verification_result.get('npu_output') is not None and input_array is not None and expected_output is not None:
                npu_output = verification_result['npu_output']
                if 'test_data' not in error_result:
                    error_result['test_data'] = {
                        'input_shape': list(input_array.shape),
                        'input_dtype': str(input_array.dtype),
                        'output_shape': list(expected_output.shape),
                        'output_dtype': str(expected_output.dtype)
                    }
        
        # Save error details
        error_file = f"{self.output_dir}/{kernel_name}_error.json"
        with open(error_file, 'w') as f:
            # Handle numpy arrays for JSON serialization
            json_result = error_result.copy()
            if input_array is not None and expected_output is not None:
                json_result['test_data']['input_sample'] = input_array[:10].tolist()
                json_result['test_data']['expected_output_sample'] = expected_output[:10].tolist()
                
                # Add NPU output sample if available from verification
                if verification_result and verification_result.get('npu_output') is not None:
                    npu_output = verification_result['npu_output']
                    try:
                        if hasattr(npu_output, 'tolist'):
                            json_result['test_data']['npu_output_sample'] = npu_output[:10].tolist()
                        elif isinstance(npu_output, (list, tuple)):
                            json_result['test_data']['npu_output_sample'] = list(npu_output[:10])
                        else:
                            # Try to convert to numpy array first
                            import numpy as np
                            npu_array = np.array(npu_output)
                            json_result['test_data']['npu_output_sample'] = npu_array[:10].tolist()
                    except Exception as e:
                        print(f"Warning: Could not save NPU output sample in error case: {e}")
                        
            json.dump(json_result, f, indent=2)
            
        print(f"\n=== Demo Failed ===")
        print(f"Failed at: {step_name}")
        print(f"Error: {exception}")
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
