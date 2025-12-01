PY=python
RUN=uvicorn src.api:app --host 0.0.0.0 --port 8000
BACKEND ?= lightgbm

prepare:
	$(PY) -m pip install -r requirements.txt

data:
	$(PY) -m src.data_prep

train:
	$(PY) -m src.train --backend $(BACKEND)

eval:
	$(PY) -m src.eval

serve:
	$(RUN)

test:
	pytest -q

.PHONY: prepare data train eval serve test