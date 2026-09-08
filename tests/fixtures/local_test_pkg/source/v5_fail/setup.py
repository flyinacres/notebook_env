from setuptools import setup
from setuptools.command.build_py import build_py as _build_py


class FailingBuildPy(_build_py):
    def run(self):
        raise RuntimeError("deliberate build failure for local_test_pkg")


setup(
    name="local_test_pkg",
    version="5.0.0",
    packages=["local_test_pkg"],
    cmdclass={"build_py": FailingBuildPy},
)
