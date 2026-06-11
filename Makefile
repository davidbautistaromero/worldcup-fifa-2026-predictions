# Atajos de desarrollo. En Windows puedes usar `make` (si lo tienes) o copiar los comandos.
.PHONY: install data train evaluate simulate dashboard test lint clean

install:
	pip install -e ".[dev,dashboard]"

data:
	worldcup download-data

train:
	worldcup train

evaluate:
	worldcup evaluate

simulate:
	worldcup simulate

dashboard:
	streamlit run dashboard/app.py

test:
	pytest

lint:
	ruff check src tests

clean:
	python -c "import shutil,glob,os; [shutil.rmtree(p,ignore_errors=True) for p in glob.glob('**/__pycache__',recursive=True)]"
