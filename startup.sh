#!/bin/bash
set -e
echo "🚀 Starting Thinking Cats Server..."
# Oryx already installed dependencies into the site root
# Activate the Oryx‑provided venv if needed (usually not required)
# exec the server
exec uvicorn api.main:app --host 0.0.0.0 --port "${PORT:-8000}"