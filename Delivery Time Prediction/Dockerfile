# Use an official lightweight Python image
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Copy all files into the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose ports
# 5000 for Flask, 8501 for Streamlit
EXPOSE 5000 

# Run Flask app by default
CMD ["python", "app.py"]
