from setuptools import setup, find_packages

# Read the README file for long description
def read_readme():
    try:
        with open("ose/README.md", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "Open Synthetic Data Engine - A library for generating synthetic data using large language models"

setup(
    name="open-synthetic-data",
    version="0.1.0",
    author="OSE Team",
    author_email="",
    description="Open Synthetic Data Engine - A library for generating synthetic data using large language models",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch",
        "numpy", 
        "tqdm",
        "transformers",
        "ray[data]",
        "jsonargparse",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "black",
            "flake8",
            "mypy",
        ],
        "docs": [
            "sphinx",
            "sphinx-rtd-theme",
        ],
    },
    entry_points={
        "console_scripts": [
            "ose-generate=ose.generate:cli",
        ],
    },
    include_package_data=True,
    package_data={
        "ose": [
            "configs/*.yaml",
            "schema/*.json",
        ],
    },
    zip_safe=False,
) 