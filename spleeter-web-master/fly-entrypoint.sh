#!/bin/bash
set -e

# Fly.io single-container entrypoint
# Runs Redis + Django + Celery via supervisor

# Set defaults for Fly.io
export CELERY_BROKER_URL="${CELERY_BROKER_URL:-redis://127.0.0.1:6379/0}"
export CELERY_RESULT_BACKEND="${CELERY_RESULT_BACKEND:-redis://127.0.0.1:6379/0}"
export API_HOST="${API_HOST:-0.0.0.0}"
export SECRET_KEY="${SECRET_KEY:-$(python3 -c 'import secrets; print(secrets.token_urlsafe(50))')}"

# Ensure dirs exist
mkdir -p /webapp/sqlite /webapp/media /webapp/celery /webapp/staticfiles

# Collect static files if assets exist
if [ -d /webapp/frontend/assets/dist ]; then
    echo "Collecting static files..."
    python3 manage.py collectstatic --noinput 2>/dev/null || true
fi

# Apply migrations
echo "Applying migrations..."
python3 manage.py migrate --noinput

echo "Starting all services via supervisor..."
exec supervisord -n -c /etc/supervisor/conf.d/fly.conf
