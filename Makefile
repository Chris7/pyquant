testenv:
	pip install -r requirements.txt
	pip install -e .

test:
	nosetests --with-coverage --cover-erase --cover-package=pyquant pyquant.tests
	coverage report

testintegration:
	nosetests -i integration* pyquant.tests

testall: test testintegration
