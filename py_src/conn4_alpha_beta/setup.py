import pybind11
import setuptools, sysconfig
from setuptools.command.build_ext import build_ext

class BuildExt(build_ext):
    def build_extensions(self):
        # Configure the C++ standard and optimization flags.
        ct = self.compiler.compiler_type
        opts = []
        if ct == 'unix':
            opts.append('-std=c++17')
            opts.append('-O3')  # Add the optimization flag here.
            opts.append('-march=native')
        elif ct == 'msvc': 
            opts.append('/EHsc')
            opts.append('/O2')  # Add the optimization flag for MSVC here.
        for ext in self.extensions:
            ext.extra_compile_args = opts
        build_ext.build_extensions(self)

setuptools.setup(
    name="conn4_solver",
    version="1",
    author="chonker",
    author_email="chonker@chungus.big",
    description="A chonky solver for Connect 4",
    ext_modules=[setuptools.Extension(
        "conn4_solver",
        ["lib.cpp","generator.cpp","Solver.cpp"],
        include_dirs=[sysconfig.get_paths()['include'], pybind11.get_include()],
        language="c++"
    )],
    cmdclass={'build_ext': BuildExt}
)

