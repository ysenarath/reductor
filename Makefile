setup:
	curl -LsSf https://gist.githubusercontent.com/ysenarath/ccadf71f7d5cd2d2c52bea974a7b33df/raw/6e5b0962644f41ac5466fd39cba0cf4adde6a29a/uv-venv-setup.sh | sh

pre-commit:
	bash .git/hooks/pre-commit

clear-cache:
	rm -rf instance/cache
	rm -rf instance/dash

build-cache:
	uv run src/socytinet/dashboard/build_cache.py

run:
	FLASK_APP=src/socytinet/dashboard/app.py FLASK_DEBUG=true flask run