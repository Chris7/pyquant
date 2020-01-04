devenv:
	pip install -r requirements-dev.txt
	pip install -e .

release/major release/minor release/patch release/test:
	bumpversion $(@F)
	git push
	git push --tags

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
