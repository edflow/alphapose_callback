from setuptools import setup, find_packages

# allows to get version via python setup.py --version
__version__ = "dev"

install_requires = [
    "pyyaml",
    "tqdm",
    "numpy",
    "pandas",  # for csv dataset and eval pipeline
    "edflow",
]

install_full = [  # for extra functionality
    "black",  # for formatting of code
    "matplotlib",  # for plot_datum
]
install_docs = [  # for building the documentation
    "sphinx >= 1.4",
    "sphinx_rtd_theme",
    "better-apidoc",
]
install_test = install_full + [  # for running the tests
    "pytest",
    "pytest-cov",
    "coveralls",
    "coverage < 5.0",  # TODO pinned dependency of coveralls; see https://github.com/coveralls-clients/coveralls-python/issues/203
]
extras_require = {"full": install_full, "test": install_test}

long_description = """Reduce boilerplate code for your ML projects. TensorFlow
and PyTorch. [Documentation](https://edflow.readthedocs.io/)"""

setup(
    name="alphapose_callback",
    version=__version__,
    description="Pose estimation and PCK calculation callback for edflow callbacks based on AlphaPose Model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/edflow/alphapose_callback",
    author="Sandro Braun",
    author_email="sandro.braun@iwr.uni-heidelberg.de",
    license="MIT",
    packages=find_packages(),
    package_data={"": ["*.yaml"]},
    install_requires=install_requires,
    extras_require=extras_require,
    zip_safe=False,
    scripts=["alphapose_callback_inference", "alphapose_callback_pck"],
    python_requires=">=3.6",
    classifiers=[
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
    ],
)
