FROM python:3.9-slim

# Install any OS packages your libraries need (for example, for scipy or audio handling)
RUN apt-get update && apt-get install -y \
    build-essential \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install them
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the add-on code into the container
COPY . /app/

# Make the run script executable
RUN chmod +x /app/run.sh

CMD [ "/app/run.sh" ]
