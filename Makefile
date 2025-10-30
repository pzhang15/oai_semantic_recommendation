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



## --- Deployment conveniences ---
.PHONY: demo-docker dev-env smoke verify clean

demo-docker:
	@if command -v docker >/dev/null 2>&1; then docker compose up --build ; \
	else ./demo.sh ; fi

dev-env:
	$(PYTHON) -m pip install -U pip
	$(PYTHON) -m pip install -r requirements.txt
	( cd frontend && npm ci || yarn || pnpm i )
	$(PYTHON) scripts/preflight.py
	$(PYTHON) scripts/bootstrap_demo.py
	uvicorn src.app:app --host $${API_HOST:-0.0.0.0} --port $${API_PORT:-8000}

smoke:
	$(PYTHON) scripts/smoke_tests.py

verify:
	./demo.sh --headless || true
	$(MAKE) smoke

clean:
	rm -rf .venv __pycache__ **/__pycache__

.PHONY: arch
arch: ## render architecture diagram (requires node dev deps)
	npm run arch || (npm install --no-audit --no-fund && npm run arch)