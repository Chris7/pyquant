testenv:
	pip install -r requirements.txt
	pip install -e .
	# download files for testing
	# mkdir -p tests/data
	# cd tests/data
	# mkdir ms3
	# mkdir silac
	# mkdir neucode
	# mkdir itraq
	# mkdir mva
	# curl http://www.ebi.ac.uk/pride/archive/files/179880101

test:
	nosetests --with-coverage --cover-erase --cover-package=pyquant pyquant.tests
	coverage report --omit=pyquant/tests*

testintegration:
	nosetests -i integration_.* pyquant.tests

testmanual:
	nosetests -i .*manual.* pyquant.tests

testtargeted:
	nosetests -i targeted_.* pyquant.tests

testall: test testintegration
