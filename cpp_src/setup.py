from setuptools import setup, Extension

setup(
    name="player",
    version="0.1",
    # ext_modules=[
    #     Extension("player", sources=[])
    # ],
    py_modules=["player"],
    # include the path to the compiled module's .so or .pyd file
    package_data={"": ["player.cpython-310-x86_64-linux-gnu.so"]},
)