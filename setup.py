from setuptools import find_packages, setup

setup(
    name='derl',
    packages=find_packages(),
    version='0.0.1',
    install_requires=[
        'setuptools==49.6.0',
        'gym',
        'click',
        'hydra-core==1.0.5',
        'omegaconf',
        'pyyaml==5.4.1',
        'torch',
        'tensorboard',
        'stable-baselines3',
        'tqdm',
        'bsuite',
    ]
)
