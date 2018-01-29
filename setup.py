import io
import sys
try:
    import pypandoc
except:
    pypandoc = None

from setuptools import find_packages, setup

with io.open('aima3/__init__.py', encoding='utf-8') as fid:
    for line in fid:
        if line.startswith('__version__'):
            version = line.strip().split()[-1][1:-1]
            break

with io.open('README.md', encoding='utf-8') as fp:
    long_desc = fp.read()
    if pypandoc is not None:
        try:
            long_desc = pypandoc.convert(long_desc, "rst", "markdown_github")
        except:
            pass

setup(name='aima3',
      version=version,
      description='Artificial Intelligence: A Modern Approach, in Python3',
      long_description=long_desc,
      author='Douglas Blank',
      author_email='doug.blank@gmail.com',
      url='https://github.com/Calysto/aima3',
      install_requires=['networkx==1.11', 'jupyter'],
      packages=find_packages(include=['aima3', 'aima3.*']),
      classifiers=[
          'Framework :: IPython',
          'License :: OSI Approved :: BSD License',
          'Programming Language :: Python :: 3',
      ]
      )
