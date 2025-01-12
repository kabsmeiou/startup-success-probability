FROM python:3.12-slim-bookworm

# Install uv package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install Docker CLI
RUN apt-get update && apt-get install -y docker.io && apt-get clean

# Copy the project into the image
ADD . /app

# Sync the project into a new environment, using the frozen lockfile
WORKDIR /app
RUN uv sync --frozen

# Set the working directory to /app/src
WORKDIR /app/src

# expose the port
EXPOSE 9696

# run the script.py file in src/ uvicorn
ENTRYPOINT ["uv", "run", "main.py"]