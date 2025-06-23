# Use official Python image
FROM python:3.10-slim


# Set working directory
WORKDIR /app

# Copy requirements
COPY app_requirements.txt .

# Install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r app_requirements.txt

# Copy the app code
COPY . .

# Expose port
EXPOSE 8000

# Run the FastAPI app with uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
