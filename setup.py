from setuptools import find_packages
from setuptools import setup


setup(name='keras_style_transfer',
      version='0.0.1',
      description='Style Transfer in Keras',
      author='Xianshun Chen',
      author_email='xs0040@gmail.com',
      url='https://github.com/chen0040/keras-style-transfer',
      download_url='https://github.com/chen0040/keras-style-transfer/tarball/0.0.1',
      license='MIT',
      install_requires=['Keras'],
      packages=find_packages())
