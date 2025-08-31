#!/bin/bash
set -e

echo "===== OpenSSL version ====="
openssl version -a

echo "===== Python SSL version ====="
python3 -c "import ssl; print(ssl.OPENSSL_VERSION)"

echo "===== Python HTTPS test ====="
python3 -c "import urllib.request; print('HTTPS test:', urllib.request.urlopen('https://www.google.com').status)"

# Install diso at runtime (will use GPU if available)
python3 -m pip install --no-cache-dir diso

# Start your API
exec python3 start_api_simple.py