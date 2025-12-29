FROM python:3.12-slim
WORKDIR /app
COPY . /app
RUN pip install -r MLProject/requirements
CMD ["python", "MLProject/modelling_tuning.py"]
