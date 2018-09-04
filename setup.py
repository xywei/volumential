#!/usr/bin/env python

from setuptools import setup, find_packages

def main():
    from setuptools import setup

    version_dict = {}
    init_filename = "volumential/version.py"
    exec(
        compile(open(init_filename, "r").read(), init_filename, "exec"),
        version_dict)

    setup(
        name='volumential',
        version=version_dict["VERSION_TEXT"],
        description="Volume potential computation powered by FMM.",
        long_description=open("README.md", "rt").read(),
        author="Xiaoyu Wei",
        author_email="wxy0516@gmail.com",
        license="MIT",
        url="",
        classifiers=[
            'Development Status :: 3 - Alpha',
            'Intended Audience :: Developers',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: MIT License',
            'Natural Language :: English',
            'Programming Language :: Python',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Information Analysis',
            'Topic :: Scientific/Engineering :: Mathematics',
            'Topic :: Scientific/Engineering :: Visualization',
            'Topic :: Software Development :: Libraries',
            'Topic :: Utilities',
        ],
        packages=find_packages(),
        install_requires=[
            "boxtree>=2013.1",
            "pytest>=2.3",
            "loo.py>=2017.1",
            "multiprocess>=0.70",
            "h5py>=2.7",
            "pyevtk>=1.1.0",
            "scipy>=1.0",
            "sumpy>=2016.1b1"
            ],
        include_package_data=True
        )


if __name__ == '__main__':
    main()
