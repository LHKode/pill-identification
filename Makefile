init:
	pip install -r requirements.txt

install-dev:
	pip install -e ./

lint:
	find . -type f -name "*.py" ! -iname ".pylintrc" | xargs pylint --rcfile=.pylintrc