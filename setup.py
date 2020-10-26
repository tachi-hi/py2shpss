#!/usr/bin/env python

from setuptools import setup, find_packages
from os import path
import re, io

def _readme():
    with open('README.rst') as readme_file:
        return readme_file.read().replace(":copyright:", "(c)")

def _requirements():
    root_dir = path.abspath(path.dirname(__file__))
    return [name.rstrip() for name in open(path.join(root_dir, 'requirements.txt')).readlines()]

def _get_version():
    version = re.search(
        r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]',  # It excludes inline comment too
        io.open('py2shpss/__init__.py', encoding='utf_8_sig').read()
        ).group(1)
    return version

requirements = _requirements()
setup_requirements = [ ]
test_requirements = _requirements()

setup(
    author="Hideyuki Tachibana",
    author_email='tachi-hi@users.noreply.github.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Topic :: Multimedia :: Sound/Audio',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="A python implementation of two-stage HPSS (a singing voice extraction method)",
    install_requires=requirements,
    license="MIT license",
    long_description=_readme(),
    # long_description_content_type='text/markdown',
    include_package_data=True,
    keywords='py2shpss',
    name='py2shpss',
    packages=find_packages(include=['py2shpss', 'py2shpss.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/tachi-hi/py2shpss',
    version=_get_version(),
    zip_safe=False,
)
