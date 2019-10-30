testenv:
	pip install -r requirements.txt
	pip install -e .

release:
	rm -f dist/*
	python setup.py sdist
	twine upload dist/*

docker-build:
	docker build -t chrismit7/pyquant .

docker-release:
	docker push chrismit7/pyquant

test:
	nosetests -I integration_.* --with-coverage --cover-erase --cover-package=pyquant tests
	coverage report --omit=tests*

testwindows:
	nosetests -I integration_.* --with-coverage --cover-erase --cover-package=pyquant -a '!linux' tests
	coverage report --omit=tests*

testintegration:
	nosetests -i integration_.* tests

testitraq:
	nosetests tests.integration_isobaric_tags

testmanual:
	nosetests tests.integration_manual

testtravis: test testitraq

testall: test testintegration
