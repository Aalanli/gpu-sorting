from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='onesweep_cuda',
    ext_modules=[
        CUDAExtension('onesweep_cuda', [
            'one_sweep_sort.cu',
            'one_sweep.cu',
        ], extra_compile_args={'nvcc': ['-arch=sm_86']})
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
) 