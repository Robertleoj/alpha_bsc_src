import setuptools, sysconfig

setuptools.setup(
    name="conn4_solver",
    version="1",
    author="chonker",
    author_email="chonker@chungus.big",
    description="A chonky solver for Connect 4",
    ext_modules=[setuptools.Extension(
        "conn4_solver",
        ["lib.cpp","generator.cpp","Solver.cpp"],
        include_dirs=[sysconfig.get_paths()['include']],
        language="c++"
    )]
)

