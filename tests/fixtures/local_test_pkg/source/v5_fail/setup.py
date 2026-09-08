from setuptools import setup
from setuptools.command.build_py import build_py as _build_py


class FailingBuildPy(_build_py):
    def run(self):
        raise RuntimeError("deliberate build failure for notebook_env test fixture")


setup(
    name="notebook_env_test_fixture",
    version="5.0.0",
    packages=["notebook_env_test_fixture"],
    cmdclass={"build_py": FailingBuildPy},
)
