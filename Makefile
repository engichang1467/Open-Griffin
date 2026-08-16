setup-env:
	sh ops/create_env.sh

install-pytorch:
	pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu130

install:
	sh ops/set_up_env.sh

test:
	python -m tests.test_griffin

# 	pip install -r requirements.txt

# 	uv pip install --index-strategy unsafe-best-match -r requirements.txt
