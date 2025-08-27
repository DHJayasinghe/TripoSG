#!/bin/bash
set -e

# Install diso at runtime (will use GPU if available)
python3.10 -m pip install --no-cache-dir diso

# Start your API
exec python3.10 start_api_simple.py