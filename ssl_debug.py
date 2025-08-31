import ssl
import urllib.request
import sys
print("Python version:", sys.version)
print("Python SSL version:", ssl.OPENSSL_VERSION)
print("System OpenSSL version:")
import subprocess
subprocess.run(["openssl", "version", "-a"])
try:
    print("HTTPS test:", urllib.request.urlopen('https://www.google.com').status)
except Exception as e:
    print("HTTPS test failed:", e)
# ...existing code...

import time
print("Debug: Sleeping for 1 hour so you can exec into the container...")
time.sleep(3600)
