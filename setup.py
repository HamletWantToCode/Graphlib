from setuptools import setup, find_packages

from Glib import __version__

setup(
    name = 'Graphlib',
    version = __version__,
    keywords=['graph neural network'],
    description = 'library for state-of-the-art graph neural networks',
    license = 'MIT License',
    url = 'https://github.com/HamletWantToCode/Graphlib.git',
    author = 'Hongbin Ren',
    author_email = 'hongbinrenscu@outlook.com',
    packages = find_packages(),
    platforms = 'any',
    py_modules= ['Glib.parse_config']
    # install_requires = [
    #     'torch-geometric',
    # ],
    classifiers = [
                   'Programming Language :: Python :: 3.6',
                  ]
)
