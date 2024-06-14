from setuptools import setup
from torch.utils import cpp_extension

cuda_max_perf_args = ["-O3", "--use_fast_math"]
setup(
    name="fast_rope",
    ext_modules=[
        cpp_extension.CUDAExtension(
            name="rope_cuda",
            sources=["csrc/rope.cu"],
            extra_compile_args={"cxx": ["-O3"], "nvcc": cuda_max_perf_args},
        )
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
