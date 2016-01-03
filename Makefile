testenv:
	pip install -r requirements.txt
	pip install -e .

test:
	nosetests --with-coverage --cover-erase --cover-package=pyquant pyquant.tests
	coverage report --omit=pyquant/tests*

testintegration:
	nosetests -i integration_.* pyquant.tests

testmanual:
	nosetests -i .*manual.* pyquant.tests

testtargeted:
	nosetests -i targeted_.* pyquant.tests

testall: test testintegration testtargeted
