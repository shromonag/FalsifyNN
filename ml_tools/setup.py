from setuptools import setup, find_packages

setup(name='ml_tools',
      install_requires=[
          'sympy',
          'Pillow'
      ],
      packages=find_packages(),
)
