FROM python:3.11-slim
WORKDIR /app
COPY train/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY train /app/train
ENV PYTHONUNBUFFERED=1
CMD ["python", "train/train.py"]