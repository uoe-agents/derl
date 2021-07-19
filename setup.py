from setuptools import find_packages, setup

setup(
    name='derl',
    packages=find_packages(),
    version='0.0.1',
    install_requires=[
        'gym',
        'click',
        'hydra-core==1.0.5',
        'omegaconf',
        'pyyaml',
        'torch',
        'tensorboard',
        'stable-baselines3',
        'tqdm',
        'bsuite',
    ]
)
