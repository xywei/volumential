#!/usr/bin/env python
# -*- encoding: utf-8 -*-

__copyright__ = "Copyright (C) 2017 Xiaoyu Wei"

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

from setuptools import setup, find_packages


def main():
    from setuptools import setup

    version_dict = {}
    init_filename = "volumential/version.py"
    exec(compile(open(init_filename, "r").read(), init_filename, "exec"), version_dict)

    setup(
        name="volumential",
        version=version_dict["VERSION_TEXT"],
        description="Volume potential computation powered by FMM.",
        long_description=open("README.md", "rt").read(),
        author="Xiaoyu Wei",
        author_email="wxy0516@gmail.com",
        license="MIT",
        url="",
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: MIT License",
            "Natural Language :: English",
            "Programming Language :: Python",
            "Topic :: Scientific/Engineering",
            "Topic :: Scientific/Engineering :: Information Analysis",
            "Topic :: Scientific/Engineering :: Mathematics",
            "Topic :: Scientific/Engineering :: Visualization",
            "Topic :: Software Development :: Libraries",
            "Topic :: Utilities",
        ],
        packages=find_packages(),
        install_requires=[
            "boxtree",
            "h5py",
            "loo.py",
            "meshmode",
            "modepy",
            "pyevtk",
            "pymbolic",
            "pytential",
            "scipy",
            "sumpy",
        ],
        extras_require={
            "transform_based_table_builder": ["multiprocess"],
            "gmsh_support": ["gmsh_interop"],
            "test": ["multiprocess", "pytest", "gmsh_interop"],
            "doc": ["sphinx", "sphinx_rtd_theme"],
        },
        include_package_data=True,
    )


if __name__ == "__main__":
    main()
