from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='onesweep_cuda',
    ext_modules=[
        CUDAExtension('onesweep2', [
            'onesweep_tch.cu',
        ], extra_compile_args={'nvcc': ['-arch=sm_86', '-lineinfo']}),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
) 