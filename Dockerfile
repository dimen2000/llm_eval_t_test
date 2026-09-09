FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY compare_reports.py .

ENTRYPOINT ["python", "compare_reports.py"]
