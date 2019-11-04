install:
	pip install . --no-deps --upgrade

build:
	python setup.py build

deploy:
	python setup.py sdist upload -r pypi

test_deploy:
	python setup.py sdist upload -r pypitest

get_version:
	@printf "v"
	@python setup.py --version

# run_tests:
# 	@bash scripts/tests.sh .
