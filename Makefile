testenv:
	pip install -r requirements.txt
	pip install -e .

release:
	rm -f dist/*
	python setup.py sdist
	twine upload dist/*

test:
	nosetests --with-coverage --cover-erase --cover-package=pyquant pyquant.tests
	coverage report --omit=pyquant/tests*

testintegration:
	nosetests -i integration_.* pyquant.tests

testitraq:
	nosetests pyquant.tests.integration_isobaric_tags

testmanual:
	nosetests pyquant.tests.integration_manual

testtravis: test testitraq

testall: test testintegration testtargeted
