from setuptools import setup, find_packages

setup(name='experiment_setup',
      install_requires=[
          'sympy',
          'numpy'
      ],
      packages=find_packages(),
)
