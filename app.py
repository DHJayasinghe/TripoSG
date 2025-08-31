import os
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Optional, List
import base64
import io

import torch
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
from PIL import Image
import trimesh

# Azure Blob Storage imports
from azure.storage.blob import BlobServiceClient, ContentSettings
from azure.core.exceptions import AzureError

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from triposg.pipelines.pipeline_triposg import TripoSGPipeline
from triposg.pipelines.pipeline_triposg_scribble import TripoSGScribblePipeline
from scripts.image_process import prepare_image
from scripts.briarmbg import BriaRMBG
from huggingface_hub import snapshot_download

# Global variables for models
triposg_pipeline = None
triposg_scribble_pipeline = None
rmbg_net = None
device = None
dtype = None

# Azure Blob Storage Configuration
AZURE_CONNECTION_STRING = os.getenv("AZURE_CONNECTION_STRING")
CONTAINER_NAME = "3dmodels"

# 3D Viewer Configuration
IMAGE_VIEWER_URL = "https://epic13prodaestorage01.z8.web.core.windows.net/?model"

# Configuration
def get_directory_path(dir_name, fallback_prefix="triposg_"):
    """Get directory path, trying current directory first, then home directory"""
    current_dir = Path.cwd() / dir_name
    home_dir = Path.home() / f"{fallback_prefix}{dir_name}"
    
    # Try current directory first
    try:
        if current_dir.exists() or os.access(Path.cwd(), os.W_OK):
            current_dir.mkdir(exist_ok=True)
            return str(current_dir)
    except (PermissionError, OSError):
        pass
    
    # Fallback to home directory
    try:
        home_dir.mkdir(exist_ok=True)
        print(f"📁 Using home directory for {dir_name}: {home_dir}")
        return str(home_dir)
    except Exception as e:
        print(f"❌ Error creating directory {dir_name}: {e}")
        # Last resort: use current directory and let it fail with clear error
        return str(current_dir)

MODEL_WEIGHTS_DIR = get_directory_path("pretrained_weights")
OUTPUT_DIR = get_directory_path("outputs")
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

print(f"📁 Final directory configuration:")
print(f"   Model weights: {MODEL_WEIGHTS_DIR}")
print(f"   Outputs: {OUTPUT_DIR}")
print(f"   Azure Container: {CONTAINER_NAME}")

# Azure Blob Storage helper functions
def get_blob_service_client():
    """Get Azure Blob Service Client"""
    try:
        return BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
    except Exception as e:
        print(f"❌ Error creating Azure Blob Service Client: {e}")
        return None

def upload_to_azure_blob(file_path: str, blob_name: str) -> Optional[str]:
    """Upload file to Azure Blob Storage and return the URL"""
    try:
        blob_service_client = get_blob_service_client()
        if not blob_service_client:
            return None
            
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        
        # Upload the file
        with open(file_path, "rb") as data:
            blob_client = container_client.get_blob_client(blob_name)
            blob_client.upload_blob(
                data, 
                overwrite=True,
                content_settings=ContentSettings(content_type="application/octet-stream")
            )
        
        # Return the blob URL
        blob_url = f"https://epic13prodaestorage01.blob.core.windows.net/{CONTAINER_NAME}/{blob_name}"
        print(f"✅ Uploaded to Azure Blob: {blob_url}")
        return blob_url
        
    except Exception as e:
        print(f"❌ Error uploading to Azure Blob: {e}")
        return None

app = FastAPI(
    title="TripoSG API",
    description="High-Fidelity 3D Shape Synthesis API using Large-Scale Rectified Flow Models",
    version="1.0.0"
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API requests/responses
class TripoSGRequest(BaseModel):
    seed: Optional[int] = 42
    num_inference_steps: Optional[int] = 50
    guidance_scale: Optional[float] = 7.0
    faces: Optional[int] = -1

class TripoSGScribbleRequest(BaseModel):
    prompt: str
    scribble_conf: Optional[float] = 0.3
    seed: Optional[int] = 42
    num_inference_steps: Optional[int] = 50
    guidance_scale: Optional[float] = 7.0
    faces: Optional[int] = -1

class TripoSGResponse(BaseModel):
    message: str
    output_file: str
    blob_url: Optional[str] = None
    viewer_url: Optional[str] = None

class TripoSGScribbleResponse(BaseModel):
    message: str
    output_file: str
    blob_url: Optional[str] = None
    viewer_url: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    device: str

def initialize_models():
    """Initialize TripoSG models and RMBG"""
    global triposg_pipeline, triposg_scribble_pipeline, rmbg_net, device, dtype
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        
        print(f"Using device: {device} with dtype: {dtype}")
        

        # Download pretrained weights only if not already present
        triposg_weights_dir = os.path.join(MODEL_WEIGHTS_DIR, "TripoSG")
        triposg_scribble_weights_dir = os.path.join(MODEL_WEIGHTS_DIR, "TripoSG-scribble")
        rmbg_weights_dir = os.path.join(MODEL_WEIGHTS_DIR, "RMBG-1.4")

        if not (os.path.exists(triposg_weights_dir) and os.listdir(triposg_weights_dir)):
            print("Downloading TripoSG weights...")
            snapshot_download(repo_id="VAST-AI/TripoSG", local_dir=triposg_weights_dir)
        else:
            print("TripoSG weights already present, skipping download.")

        if not (os.path.exists(triposg_scribble_weights_dir) and os.listdir(triposg_scribble_weights_dir)):
            print("Downloading TripoSG-scribble weights...")
            snapshot_download(repo_id="VAST-AI/TripoSG-scribble", local_dir=triposg_scribble_weights_dir)
        else:
            print("TripoSG-scribble weights already present, skipping download.")

        if not (os.path.exists(rmbg_weights_dir) and os.listdir(rmbg_weights_dir)):
            print("Downloading RMBG weights...")
            snapshot_download(repo_id="briaai/RMBG-1.4", local_dir=rmbg_weights_dir)
        else:
            print("RMBG weights already present, skipping download.")
        
        # Initialize RMBG model
        print("Initializing RMBG model...")
        rmbg_net = BriaRMBG.from_pretrained(rmbg_weights_dir).to(device)
        rmbg_net.eval()
        
        # Initialize TripoSG pipeline
        print("Initializing TripoSG pipeline...")
        triposg_pipeline = TripoSGPipeline.from_pretrained(triposg_weights_dir).to(device, dtype)
        
        # Initialize TripoSG-scribble pipeline
        print("Initializing TripoSG-scribble pipeline...")
        triposg_scribble_pipeline = TripoSGScribblePipeline.from_pretrained(triposg_scribble_weights_dir).to(device, dtype)
        
        print("All models initialized successfully!")
        return True
        
    except Exception as e:
        print(f"Error initializing models: {e}")
        return False

def save_upload_file_tmp(upload_file: UploadFile) -> str:
    """Save uploaded file to temporary location"""
    try:
        suffix = Path(upload_file.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            content = upload_file.file.read()
            if len(content) > MAX_FILE_SIZE:
                raise HTTPException(status_code=400, detail="File too large")
            tmp_file.write(content)
            tmp_file.flush()
            return tmp_file.name
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")

def simplify_mesh(mesh: trimesh.Trimesh, n_faces: int) -> trimesh.Trimesh:
    """Simplify mesh to target number of faces"""
    if n_faces > 0 and mesh.faces.shape[0] > n_faces:
        try:
            import pymeshlab
            mesh_pymesh = pymeshlab.Mesh(vertex_matrix=mesh.vertices, face_matrix=mesh.faces)
            ms = pymeshlab.MeshSet()
            ms.add_mesh(mesh_pymesh)
            ms.meshing_merge_close_vertices()
            ms.meshing_decimation_quadric_edge_collapse(targetfacenum=n_faces)
            simplified_mesh = ms.current_mesh()
            return trimesh.Trimesh(vertices=simplified_mesh.vertex_matrix(), 
                                   faces=simplified_mesh.face_matrix())
        except ImportError:
            print("pymeshlab not available, returning original mesh")
            return mesh
    return mesh

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    print("Starting TripoSG API...")
    success = initialize_models()
    if not success:
        print("Warning: Models failed to initialize. API may not work properly.")

@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    models_loaded = (triposg_pipeline is not None and 
                    triposg_scribble_pipeline is not None and 
                    rmbg_net is not None)
    
    return HealthResponse(
        status="healthy",
        models_loaded=models_loaded,
        device=device or "unknown"
    )

@app.post("/generate-3d", response_model=TripoSGResponse)
async def generate_3d_from_image(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(None),
    image_url: str = None,
    seed: int = 42,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.0,
    faces: int = -1
):
    """Generate 3D mesh from uploaded image or image URL"""
    if triposg_pipeline is None:
        raise HTTPException(status_code=503, detail="TripoSG model not loaded")
    
    try:
        temp_file_path = None
        
        # Handle image upload or URL
        if image:
            # Validate file type
            if not image.content_type.startswith('image/'):
                raise HTTPException(status_code=400, detail="File must be an image")
            
            # Save uploaded file
            temp_file_path = save_upload_file_tmp(image)
        elif image_url:
            # Download image from URL
            import requests
            try:
                response = requests.get(image_url, timeout=30)
                response.raise_for_status()
                
                # Create temporary file
                suffix = Path(image_url).suffix or '.png'
                if not suffix.startswith('.'):
                    suffix = '.' + suffix
                
                temp_file_path = tempfile.mktemp(suffix=suffix)
                with open(temp_file_path, 'wb') as f:
                    f.write(response.content)
                    
                print(f"✅ Downloaded image from URL: {image_url}")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to download image from URL: {str(e)}")
        else:
            raise HTTPException(status_code=400, detail="Either image file or image_url must be provided")
        
        # Generate unique output filename
        output_filename = f"triposg_output_{uuid.uuid4().hex[:8]}.obj"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        # Process image and generate 3D
        img_pil = prepare_image(temp_file_path, bg_color=np.array([1.0, 1.0, 1.0]), rmbg_net=rmbg_net)
        
        with torch.no_grad():
            outputs = triposg_pipeline(
                image=img_pil,
                generator=torch.Generator(device=device).manual_seed(seed),
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            ).samples[0]
        
        # Create mesh
        mesh = trimesh.Trimesh(outputs[0].astype(np.float32), np.ascontiguousarray(outputs[1]))
        
        # Simplify mesh if requested
        if faces > 0:
            mesh = simplify_mesh(mesh, faces)
        
        # Export mesh
        mesh.export(output_path)
        
        # Upload to Azure Blob Storage
        azure_blob_url = None
        try:
            blob_name = f"triposg/{output_filename}"
            azure_blob_url = upload_to_azure_blob(output_path, blob_name)
            print(f"✅ OBJ file uploaded to Azure: {blob_name}")
        except Exception as e:
            print(f"⚠️  Azure upload failed: {e}")
        
        # Clean up temp file
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        
        # Clean up generated OBJ file after Azure upload
        if os.path.exists(output_path):
            os.unlink(output_path)
            print(f"✅ Cleaned up local OBJ file: {output_path}")
        
        return TripoSGResponse(
            message="3D mesh generated successfully",
            output_file=output_filename,
            blob_url=azure_blob_url,
            viewer_url=f"{IMAGE_VIEWER_URL}={azure_blob_url}" if azure_blob_url else None
        )
        
    except Exception as e:
        # Clean up temp file on error
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Error generating 3D: {str(e)}")

@app.post("/generate-3d-scribble", response_model=TripoSGResponse)
async def generate_3d_from_scribble(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    prompt: str = "",
    scribble_conf: float = 0.3,
    seed: int = 42,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.0,
    faces: int = -1
):
    """Generate 3D mesh from scribble image and prompt"""
    if triposg_scribble_pipeline is None:
        raise HTTPException(status_code=503, detail="TripoSG-scribble model not loaded")
    
    try:
        # Validate file type
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt is required for scribble generation")
        
        # Save uploaded file
        temp_file_path = save_upload_file_tmp(image)
        
        # Generate unique output filename
        output_filename = f"triposg_scribble_{uuid.uuid4().hex[:8]}.obj"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        # Process image and generate 3D
        img_pil = prepare_image(temp_file_path, bg_color=np.array([1.0, 1.0, 1.0]), rmbg_net=rmbg_net)
        
        with torch.no_grad():
            outputs = triposg_scribble_pipeline(
                image=img_pil,
                prompt=prompt,
                scribble_conf=scribble_conf,
                generator=torch.Generator(device=device).manual_seed(seed),
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            ).samples[0]
        
        # Create mesh
        mesh = trimesh.Trimesh(outputs[0].astype(np.float32), np.ascontiguousarray(outputs[1]))
        
        # Simplify mesh if requested
        if faces > 0:
            mesh = simplify_mesh(mesh, faces)
        
        # Export mesh
        mesh.export(output_path)
        
        # Upload to Azure Blob Storage
        azure_blob_url = None
        try:
            blob_name = f"triposg_scribble/{output_filename}"
            azure_blob_url = upload_to_azure_blob(output_path, blob_name)
            print(f"✅ OBJ file uploaded to Azure: {blob_name}")
        except Exception as e:
            print(f"⚠️  Azure upload failed: {e}")
        
        # Clean up temp file
        os.unlink(temp_file_path)
        
        # Clean up generated OBJ file after Azure upload
        if os.path.exists(output_path):
            os.unlink(output_path)
            print(f"✅ Cleaned up local OBJ file: {output_path}")
        
        return TripoSGResponse(
            message="3D mesh generated successfully from scribble",
            output_file=output_filename,
            blob_url=azure_blob_url,
            viewer_url=f"{IMAGE_VIEWER_URL}={azure_blob_url}" if azure_blob_url else None
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating 3D from scribble: {str(e)}")

@app.get("/download/{filename}")
async def download_file(filename: str):
    """Download generated 3D file"""
    file_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="application/octet-stream"
    )

@app.get("/models")
async def list_models():
    """List available models and their status"""
    return {
        "triposg": triposg_pipeline is not None,
        "triposg_scribble": triposg_scribble_pipeline is not None,
        "rmbg": rmbg_net is not None,
        "device": device,
        "dtype": str(dtype) if dtype else None
    }

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
