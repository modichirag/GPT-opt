from setuptools import setup, find_packages

setup(
    name='gptopt',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'setuptools-rust',
        'cython',
        'torch',
        'pandas',
        'matplotlib',
        'scikit-learn',
        'requests',
        'transformers',
        'datasets',
        'accelerate',
        'tiktoken',
        'zstandard'
    ],
)
