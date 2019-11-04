#!/usr/bin/env python
from pathlib import Path
from setuptools import setup, find_packages

try: # for pip >= 10
	from pip._internal.req import parse_requirements
except ImportError: # for pip <= 9.0.3
	from pip.req import parse_requirements

pkg_name = "cluster_parts"

cwd = Path(__file__).parent.resolve()
# Get __version__ variable
exec(open(cwd / pkg_name / '_version.py').read())

install_requires = [
	line.strip()
		for line in open(cwd / "requirements.txt").readlines()
]

setup(
	name=pkg_name,
	version=__version__,
	description='',
	author='Dimitri Korsch',
	author_email='korschdima@gmail.com',
	license='MIT License',
	packages=find_packages(),
	zip_safe=False,
	setup_requires=[],
	install_requires=install_requires,
	package_data={'': ['requirements.txt']},
	data_files=[('.',['requirements.txt'])],
	include_package_data=True,
)
