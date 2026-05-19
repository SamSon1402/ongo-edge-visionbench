.PHONY: help install test lint type bench api docker clean

PYTHON := python3.11
PKG    := ongoedge

help:
	@echo "ongoedge — edge cv benchmark suite"
	@echo ""
	@echo "  install     install package + dev deps in editable mode"
	@echo "  test        run pytest with coverage"
	@echo "  lint        ruff check + format"
	@echo "  type        mypy strict"
	@echo "  bench       run a host-cpu smoke sweep"
	@echo "  api         launch fastapi dashboard backend"
	@echo "  docker      build jetson-targeted image"
	@echo "  clean       remove build / cache artifacts"

install:
	$(PYTHON) -m pip install -e ".[dev,onnx]"

test:
	$(PYTHON) -m pytest --cov=$(PKG) --cov-report=term-missing

lint:
	$(PYTHON) -m ruff check src tests
	$(PYTHON) -m ruff format --check src tests

type:
	$(PYTHON) -m mypy src/$(PKG)

bench:
	$(PYTHON) -m $(PKG).cli run \
		--target host-cpu \
		--models yolo11n,mobilevit-xs \
		--precision fp16 \
		--frames 100

api:
	$(PYTHON) -m uvicorn $(PKG).api:app --host 0.0.0.0 --port 8000 --reload

docker:
	docker buildx build \
		--platform linux/arm64 \
		-t ongoedge/bench:jetson-orin \
		-f docker/Dockerfile.jetson .

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
