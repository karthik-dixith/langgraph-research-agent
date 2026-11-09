FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt pyproject.toml ./
COPY harness ./harness
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir -e .

COPY api.py ./

EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
