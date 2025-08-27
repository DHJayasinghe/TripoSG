#!/usr/bin/env python3
"""
Test script for TripoSG API endpoints
Run this after starting the API server to test functionality
"""

import requests
import json
import time
from pathlib import Path

API_BASE = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint"""
    print("🔍 Testing health check...")
    try:
        response = requests.get(f"{API_BASE}/")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data}")
            return data.get('models_loaded', False)
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Make sure the server is running on localhost:8000")
        return False

def test_models_status():
    """Test the models status endpoint"""
    print("\n🔍 Testing models status...")
    try:
        response = requests.get(f"{API_BASE}/models")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Models status: {json.dumps(data, indent=2)}")
            return data
        else:
            print(f"❌ Models status failed: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Error testing models status: {e}")
        return None

def test_api_documentation():
    """Test if API documentation is accessible"""
    print("\n🔍 Testing API documentation...")
    try:
        response = requests.get(f"{API_BASE}/docs")
        if response.status_code == 200:
            print("✅ API documentation accessible")
            return True
        else:
            print(f"❌ API documentation not accessible: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error testing API documentation: {e}")
        return False

def test_static_files():
    """Test if static files are accessible"""
    print("\n🔍 Testing static files...")
    try:
        response = requests.get(f"{API_BASE}/static/index.html")
        if response.status_code == 200:
            print("✅ Static files accessible")
            return True
        else:
            print(f"❌ Static files not accessible: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error testing static files: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Testing TripoSG API...")
    print("=" * 50)
    
    # Test basic connectivity
    if not test_health_check():
        print("\n❌ API is not ready. Please wait for models to load or check server status.")
        return
    
    # Test models status
    models_status = test_models_status()
    
    # Test API documentation
    test_api_documentation()
    
    # Test static files
    test_static_files()
    
    print("\n" + "=" * 50)
    if models_status:
        print("🎉 All tests passed! API is ready to use.")
        print("\n📱 You can now:")
        print("   - Use the web interface at: http://localhost:8000/static/index.html")
        print("   - View API docs at: http://localhost:8000/docs")
        print("   - Make API calls to: http://localhost:8000")
    else:
        print("⚠️  API is running but models are still loading.")
        print("   Please wait a few minutes and run the test again.")

if __name__ == "__main__":
    main()
