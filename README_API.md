# TripoSG API Service

A FastAPI-based web service that provides REST API endpoints for TripoSG 3D shape generation, similar to your TripoSR project's localhost functionality.

## 🚀 Features

- **REST API**: Full REST API for 3D generation from images
- **Web Interface**: Beautiful HTML frontend for easy testing
- **Two Generation Modes**:
  - **Image to 3D**: Generate 3D meshes from regular images
  - **Scribble to 3D**: Generate 3D meshes from scribble drawings + text prompts
- **Automatic Model Management**: Downloads and manages model weights automatically
- **File Management**: Handles file uploads and provides download links
- **Real-time Status**: Health checks and model loading status

## 📁 Project Structure

```
TripoSG-main/
├── app.py                 # Main FastAPI application
├── start_api.py          # Startup script with dependency checks
├── test_api.py           # API testing script
├── requirements_api.txt   # API-specific dependencies
├── static/               # Web interface files
│   └── index.html        # HTML frontend
├── outputs/              # Generated 3D files (created automatically)
├── pretrained_weights/   # Model weights (downloaded automatically)
└── triposg/              # Original TripoSG library
```

## 🛠️ Installation

### 1. Install API Dependencies

```bash
pip install -r requirements_api.txt
```

### 2. Verify Installation

```bash
python start_api.py --check
```

## 🚀 Quick Start

### Option 1: Use Startup Script (Recommended)

```bash
python start_api.py
```

### Option 2: Direct Start

```bash
python app.py
```

The server will start on `http://localhost:8000`

## 🌐 API Endpoints

### Health Check
- **GET** `/` - Check API health and model status

### 3D Generation
- **POST** `/generate-3d` - Generate 3D from image
- **POST** `/generate-3d-scribble` - Generate 3D from scribble + prompt

### File Management
- **GET** `/download/{filename}` - Download generated 3D files

### System Info
- **GET** `/models` - Check model loading status

### Web Interface
- **GET** `/static/index.html` - Web-based UI for testing

## 📱 Web Interface

Access the web interface at: `http://localhost:8000/static/index.html`

Features:
- **Tabbed Interface**: Switch between Image→3D and Scribble→3D modes
- **Image Preview**: See uploaded images before processing
- **Parameter Control**: Adjust generation parameters
- **Real-time Status**: See generation progress and results
- **Download Links**: Direct download of generated 3D files

## 🔧 API Usage Examples

### Generate 3D from Image

```python
import requests

# Upload image and generate 3D
with open('image.jpg', 'rb') as f:
    files = {'image': f}
    data = {
        'seed': 42,
        'num_inference_steps': 50,
        'guidance_scale': 7.0,
        'faces': 5000
    }
    
    response = requests.post(
        'http://localhost:8000/generate-3D',
        files=files,
        data=data
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"Generated: {result['output_file']}")
        print(f"Download: http://localhost:8000{result['download_url']}")
```

### Generate 3D from Scribble

```python
import requests

# Upload scribble and generate 3D
with open('scribble.png', 'rb') as f:
    files = {'image': f}
    data = {
        'prompt': 'a cat with wings',
        'scribble_conf': 0.3,
        'seed': 42,
        'num_inference_steps': 50,
        'guidance_scale': 7.0,
        'faces': 5000
    }
    
    response = requests.post(
        'http://localhost:8000/generate-3d-scribble',
        files=files,
        data=data
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"Generated: {result['output_file']}")
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file for custom configuration:

```env
# Server Configuration
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=info

# Model Configuration
MODEL_WEIGHTS_DIR=pretrained_weights
OUTPUT_DIR=outputs
MAX_FILE_SIZE=52428800  # 50MB in bytes

# CUDA Configuration
FORCE_CPU=false
```

### Model Parameters

- **seed**: Random seed for reproducible results (default: 42)
- **num_inference_steps**: Number of denoising steps (default: 50, range: 1-100)
- **guidance_scale**: How closely to follow the input (default: 7.0, range: 0.1-20.0)
- **faces**: Maximum number of faces in output mesh (default: -1, no limit)
- **scribble_conf**: Confidence in scribble input (default: 0.3, range: 0.0-1.0)

## 🧪 Testing

### Test API Endpoints

```bash
python test_api.py
```

### Manual Testing

1. Start the server: `python start_api.py`
2. Open web interface: `http://localhost:8000/static/index.html`
3. Upload an image and test generation
4. Check API docs: `http://localhost:8000/docs`

## 📊 Performance

### Model Sizes
- **TripoSG**: ~1.5GB (main model)
- **TripoSG-scribble**: ~512MB (scribble model)
- **RMBG**: ~100MB (background removal)

### Generation Time
- **GPU (CUDA)**: 2-5 minutes per generation
- **CPU**: 10-30 minutes per generation

### Memory Requirements
- **GPU VRAM**: Minimum 8GB recommended
- **System RAM**: Minimum 16GB recommended

## 🔍 Troubleshooting

### Common Issues

1. **Models not loading**
   - Check internet connection for model downloads
   - Verify CUDA installation if using GPU
   - Check available disk space

2. **CUDA errors**
   - Install compatible PyTorch version
   - Check CUDA driver compatibility
   - Use CPU mode as fallback

3. **Memory errors**
   - Reduce batch size or image resolution
   - Close other applications
   - Use model quantization

### Logs and Debugging

Enable debug logging:

```bash
LOG_LEVEL=debug python app.py
```

Check model status:

```bash
curl http://localhost:8000/models
```

## 🔒 Security Considerations

- **File Upload Limits**: 50MB maximum file size
- **CORS**: Configured for development (allows all origins)
- **Input Validation**: File type and size validation
- **Temporary Files**: Automatic cleanup of uploaded files

## 📈 Scaling and Production

### For Production Use

1. **Environment Variables**: Set proper production values
2. **Reverse Proxy**: Use nginx or Apache
3. **Process Management**: Use systemd or supervisor
4. **Monitoring**: Add logging and metrics
5. **Authentication**: Implement API keys or OAuth

### Load Balancing

```bash
# Multiple instances
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Test thoroughly
5. Submit pull request

## 📄 License

This API service follows the same license as the TripoSG project.

## 🙏 Acknowledgments

- **TripoSG Team**: For the amazing 3D generation models
- **FastAPI**: For the excellent web framework
- **Hugging Face**: For model hosting and distribution

## 📞 Support

- **Issues**: Use GitHub Issues
- **Documentation**: Check `/docs` endpoint when server is running
- **Community**: Join TripoSG discussions

---

**Happy 3D Generating! 🎉**
