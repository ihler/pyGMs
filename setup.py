import setuptools 

#import pyGM

with open("readme.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='pyGMs',
    #name=pyGM.__title__,
    #version=pyGM.__version__,
    version='0.1.1',
    author='Alexander Ihler',
    author_email='ihler@ics.uci.edu',
    description='Python Graphical Models Toolbox',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
    ],
)

