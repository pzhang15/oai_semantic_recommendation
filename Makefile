PYTHON ?= python

.PHONY: init run index demo embed build-index index-all diagrams docs

init:
	@echo "Initialize environment and install dependencies"
	$(PYTHON) -m pip install -U pip
	$(PYTHON) -m pip install -r requirements.txt

run:
	uvicorn src.app:app --reload

index:
	$(PYTHON) scripts/index.py

demo:
	$(PYTHON) scripts/index.py --demo

embed:
	$(PYTHON) scripts/index.py --embed

build-index:
	$(PYTHON) scripts/index.py --build-faiss

index-all:
	$(PYTHON) scripts/index.py --embed --build-faiss


diagrams:
	$(PYTHON) scripts/render_diagrams.py

docs: diagrams


