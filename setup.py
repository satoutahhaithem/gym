from setuptools import setup, find_packages

# Read core dependencies from requirements.txt
with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()
    # Filter out comments and empty lines
    requirements = [line for line in requirements if line.strip() and not line.startswith('#')]

# Optional dependencies
extras_require = {
    'wandb': ['wandb>=0.12.0'],
    's3': ['boto3>=1.20.0'], 
    'demo': ['einops>=0.6.0'],
    'examples': ['torchvision>=0.15.0'],
    'dev': [
        'wandb>=0.12.0',
        'boto3>=1.20.0',
        'einops>=0.6.0',
        'torchvision>=0.15.0',
        'pytest>=7.0.0',
        'black>=22.0.0',
        'flake8>=4.0.0',
        'mypy>=0.950'
    ],
    'all': [
        'wandb>=0.12.0',
        'boto3>=1.20.0',
        'einops>=0.6.0',
        'torchvision>=0.15.0'
    ]
}

setup(
    name="DistributedSim",
    version="0.1.0",
    description="Distributed Training Simulator",
    author="Matt Beton",
    author_email="matthew.beton@gmail.com",
    url="https://github.com/MattyAB/DistributedSim",
    packages=find_packages(),  # Automatically find packages in the directory
    install_requires=requirements,  # Install core dependencies from requirements.txt
    extras_require=extras_require,  # Optional dependencies
    python_requires=">=3.8",  # Specify Python version compatibility
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="distributed training, machine learning, deep learning",
    license="MIT",
)