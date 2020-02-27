FROM python:3.7

RUN python -m pip install --upgrade pip setuptools
COPY cosine_model.py cosine_model.py
COPY requirements.txt requirements.txt
RUN python -m pip install -r requirements.txt
ENTRYPOINT ["python", "cosine_model.py"]