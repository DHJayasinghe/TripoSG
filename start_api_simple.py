#!/usr/bin/env python3
"""
Simplified TripoSG API Startup Script for Ubuntu VM
This script handles permission issues and creates directories safely
"""

import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import torch
        import fastapi
        import uvicorn
        print("✅ All required dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install dependencies with: pip install -r requirements_api.txt")
        return False

def check_cuda():
    """Check CUDA availability"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
            return True
        else:
            print("⚠️  CUDA not available, will use CPU (slower)")
            return False
    except Exception as e:
        print(f"⚠️  Could not check CUDA: {e}")
        return False

def create_directories_safe():
    """Create directories safely, handling permission issues"""
    print("📁 Creating directories safely...")
    
    # Get current working directory and home directory
    current_dir = Path.cwd()
    home_dir = Path.home()
    
    dirs_to_create = ["outputs", "pretrained_weights", "static"]
    created_dirs = {}
    
    for dir_name in dirs_to_create:
        # Try current directory first
        current_path = current_dir / dir_name
        try:
            current_path.mkdir(exist_ok=True)
            created_dirs[dir_name] = str(current_path)
            print(f"✅ Created: {current_path}")
        except PermissionError:
            # Try home directory
            home_path = home_dir / f"triposg_{dir_name}"
            try:
                home_path.mkdir(exist_ok=True)
                created_dirs[dir_name] = str(home_path)
                print(f"✅ Created in home: {home_path}")
                
                # Create symlink if possible
                try:
                    if current_path.exists():
                        if current_path.is_symlink():
                            current_path.unlink()
                        else:
                            print(f"⚠️  {current_path} exists but is not a symlink, skipping symlink creation")
                            continue
                    current_path.symlink_to(home_path)
                    print(f"✅ Created symlink: {current_path} -> {home_path}")
                except Exception as e:
                    print(f"⚠️  Could not create symlink: {e}")
                    print(f"   Will use home directory path directly")
                    
            except Exception as e:
                print(f"❌ Could not create {dir_name} anywhere: {e}")
                return False
    
    return True

def main():
    """Main startup function"""
    print("🚀 Starting TripoSG API Server (Ubuntu VM Edition)...")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check CUDA
    check_cuda()
    
    # Create directories safely
    if not create_directories_safe():
        print("\n❌ Failed to create necessary directories.")
        print("💡 Try running the fix script:")
        print("   chmod +x fix_permissions.sh")
        print("   ./fix_permissions.sh")
        sys.exit(1)
    
    print("\n📁 Directory structure:")
    print("   ├── app.py              # Main FastAPI application")
    print("   ├── static/             # Web interface files")
    print("   ├── outputs/            # Generated 3D files")
    print("   └── pretrained_weights/ # Model weights (downloaded automatically)")
    
    print("\n🌐 Starting server...")
    print("   - API will be available at: http://localhost:8000")
    print("   - Web interface at: http://localhost:8000/static/index.html")
    print("   - API documentation at: http://localhost:8000/docs")
    
    print("\n⏳ Initializing models (this may take several minutes on first run)...")
    print("   - TripoSG model (~1.5GB)")
    print("   - TripoSG-scribble model (~512MB)")
    print("   - RMBG background removal model (~100MB)")
    
    # Start the server
    try:
        subprocess.run([sys.executable, "app.py"], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Server failed to start: {e}")
        print("\n💡 Troubleshooting tips:")
        print("   1. Check if port 8000 is available: netstat -tlnp | grep :8000")
        print("   2. Try different port: PORT=8001 python app.py")
        print("   3. Check permissions: ls -la")
        sys.exit(1)

if __name__ == "__main__":
    main()
