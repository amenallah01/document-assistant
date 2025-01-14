# Use an official Python base image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the project files into the container
COPY . /app

# Install required Python libraries
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port for communication (if needed, e.g., API or UI)
EXPOSE 8000

# Set the default command to run the program
CMD ["python", "src/main.py"]
