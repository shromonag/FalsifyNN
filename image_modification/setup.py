from setuptools import setup, find_packages

setup(name='image_modification',
      install_requires=[
          'sympy',
          'Pillow'
      ],
      packages=find_packages(),
)
