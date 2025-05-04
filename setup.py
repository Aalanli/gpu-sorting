from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='onesweep_cuda',
    ext_modules=[
        CUDAExtension('onesweep_cuda', [
            'one_sweep_sort.cu',
            'one_sweep.cu',
        ], extra_compile_args={'nvcc': ['-arch=sm_86', '-lineinfo']}),
        CUDAExtension("sweep", [
            'coalesced_sweep.cu',
            'coalesced_sweep_tch.cc',
        ], include_dirs=['.'], extra_compile_args={'nvcc': ['-arch=sm_86', '-lineinfo']}),
        CUDAExtension('onesweep2', [
            'one_sweep_tch.cu',
            'one_sweep.cu',
        ], extra_compile_args={'nvcc': ['-arch=sm_86', '-lineinfo']}),
        
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
) 