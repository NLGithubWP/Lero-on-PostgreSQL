# Use an official Ubuntu 22.04 LTS as a parent image
FROM ubuntu:22.04

# Set the maintainer label
LABEL maintainer="yourname@example.com"

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install certificates, gnupg, and manually add GPG keys
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    gnupg \
    && apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 3B4FE6ACC0B21F32 && \
    apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 871920D1991BC93C && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Update and install any needed packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    libreadline-dev \
    zlib1g-dev \
    vim \
    wget \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /usr/src/app

# Command to run on container start
CMD ["tail", "-f", "/dev/null"]
