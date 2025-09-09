import setuptools 

with open("readme.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='pyGMs',
    version='0.3.4',
    author='Alexander Ihler',
    author_email='ihler@ics.uci.edu',
    description='Python Graphical Models Toolbox',
    long_description=long_description,
    long_description_content_type="text/markdown",
    include_package_data=True, 
    package_data={
        'pyGMs.data': ['sources.json'],
    },
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
    ],
)

