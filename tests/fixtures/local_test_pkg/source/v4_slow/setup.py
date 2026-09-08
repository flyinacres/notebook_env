import time

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py


class SlowBuildPy(_build_py):
    def run(self):
        time.sleep(5)
        super().run()


setup(
    name="local_test_pkg",
    version="4.0.0",
    packages=["local_test_pkg"],
    cmdclass={"build_py": SlowBuildPy},
)
