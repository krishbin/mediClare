# Use an official Python runtime as a parent image
FROM python:3.10

# Set the working directory in the container
WORKDIR /backend

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

RUN pip install git+https://github.com/openai/CLIP.git

# Bundle your app's source code inside the Docker image
COPY . .

# Expose port 8000 to the outside world
EXPOSE 8000

# Command to run your FastAPI application
# CMD ["python", "main.py"]
CMD ["uvicorn", "main:imagesearch", "--host", "0.0.0.0", "--port", "8000"]