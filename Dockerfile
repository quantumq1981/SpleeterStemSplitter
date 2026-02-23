FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN mkdir -p /webapp/media /webapp/staticfiles /webapp/sqlite
WORKDIR /webapp

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ffmpeg git libasound2-dev libsndfile-dev libhdf5-dev \
    libmagic-dev redis-server supervisor nodejs npm \
    && rm -rf /var/lib/apt/lists/*

# For yt-dlp
COPY --from=denoland/deno:bin-2.5.6 /deno /usr/local/bin/deno

# Install Python dependencies (lightweight — no torch/tensorflow/demucs)
COPY spleeter-web-master/requirements-fly.txt /webapp/requirements.txt
RUN pip install --upgrade pip wheel \
    && pip install --no-cache-dir -r requirements.txt

# Copy the full application
COPY spleeter-web-master/ /webapp/

# Build frontend assets
RUN cd /webapp/frontend && npm ci && npm run build \
    && rm -rf node_modules

# Clean up node to save image space
RUN apt-get purge -y nodejs npm && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Supervisor config
COPY spleeter-web-master/fly-supervisor.conf /etc/supervisor/conf.d/fly.conf

# Entrypoint
COPY spleeter-web-master/fly-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/fly-entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["fly-entrypoint.sh"]
