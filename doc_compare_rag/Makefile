.PHONY: install run test lint

install:
	pip install -r requirements.txt

run:
	streamlit run app/ui/streamlit_app.py

test:
	pytest tests/ -v

lint:
	flake8 app/ tests/
