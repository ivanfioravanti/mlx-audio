import os
import uuid
import argparse
from fastapi import FastAPI, Response, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import soundfile as sf
import numpy as np
import pkg_resources
import importlib.util
import sys

# Import from mlx_audio package
from mlx_audio.tts.utils import load_model
from mlx_audio.tts.generate import main as generate_main

app = FastAPI()

# Add CORS middleware to allow requests from the same origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins, will be restricted by host binding
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model once on server startup.
# You can change the model path or pass arguments as needed.
# For performance, load once globally:
MODEL_PATH = "prince-canuma/Kokoro-82M"
tts_model = None  # Will be loaded when the server starts

# Make sure the output folder for generated TTS files exists
OUTPUT_FOLDER = os.path.join(os.path.expanduser("~"), ".mlx_audio", "outputs")


@app.post("/tts")
def tts_endpoint(text: str = Form(...), voice: str = Form("af_heart"), speed: float = Form(1.0)):
    """
    POST an x-www-form-urlencoded form with 'text' (and optional 'voice' and 'speed').
    We run TTS on the text, save the audio in a unique file,
    and return JSON with the filename so the client can retrieve it.
    """
    global tts_model
    
    if not text.strip():
        return JSONResponse({"error": "Text is empty"}, status_code=400)

    # Validate speed parameter
    try:
        speed_float = float(speed)
        if speed_float < 0.5 or speed_float > 2.0:
            return JSONResponse({"error": "Speed must be between 0.5 and 2.0"}, status_code=400)
    except ValueError:
        return JSONResponse({"error": "Invalid speed value"}, status_code=400)

    # We'll do something like the code in model.generate() from the TTS library:
    # Generate the unique filename
    unique_id = str(uuid.uuid4())
    filename = f"tts_{unique_id}.wav"
    output_path = os.path.join(OUTPUT_FOLDER, filename)

    # We'll use the high-level "model.generate" method:
    results = tts_model.generate(
        text=text,
        voice=voice,
        speed=speed_float,
        lang_code="a",
        verbose=False,
    )

    # We'll just gather all segments (if any) into a single wav
    # It's typical for multi-segment text to produce multiple wave segments:
    audio_arrays = []
    for segment in results:
        audio_arrays.append(segment.audio)

    # If no segments, return error
    if not audio_arrays:
        return JSONResponse({"error": "No audio generated"}, status_code=500)

    # Concatenate all segments
    cat_audio = np.concatenate(audio_arrays, axis=0)

    # Write the audio as a WAV
    sf.write(output_path, cat_audio, 24000)

    return {"filename": filename}


@app.get("/audio/{filename}")
def get_audio_file(filename: str):
    """
    Return an audio file from the outputs folder.
    The user can GET /audio/<filename> to fetch the WAV file.
    """
    file_path = os.path.join(OUTPUT_FOLDER, filename)
    if not os.path.exists(file_path):
        return JSONResponse({"error": "File not found"}, status_code=404)
    return FileResponse(file_path, media_type="audio/wav")


@app.get("/")
def root():
    """
    Serve the audio_player.html page or a fallback HTML if not found
    """
    try:
        # Try to find the audio_player.html file in the package
        static_dir = find_static_dir()
        audio_player_path = os.path.join(static_dir, "audio_player.html")
        
        if os.path.exists(audio_player_path):
            return FileResponse(audio_player_path)
        else:
            # If audio_player.html is not found, return a simple HTML page
            return HTMLResponse(content=get_fallback_html(), status_code=200)
    except Exception as e:
        # If there's an error, return a simple HTML page with error information
        return HTMLResponse(
            content=f"""
            <html>
                <head><title>MLX-Audio TTS Server</title></head>
                <body>
                    <h1>MLX-Audio TTS Server</h1>
                    <p>The server is running, but the web interface could not be loaded.</p>
                    <p>Error: {str(e)}</p>
                    <h2>API Endpoints</h2>
                    <ul>
                        <li><code>POST /tts</code> - Generate TTS audio</li>
                        <li><code>GET /audio/{{filename}}</code> - Retrieve generated audio file</li>
                    </ul>
                </body>
            </html>
            """,
            status_code=200
        )


def find_static_dir():
    """Find the static directory containing HTML files."""
    # Try different methods to find the static directory
    
    # Method 1: Use importlib.resources (Python 3.9+)
    try:
        import importlib.resources as pkg_resources
        static_dir = pkg_resources.files('mlx_audio').joinpath('tts')
        static_dir_str = str(static_dir)
        if os.path.exists(static_dir_str):
            return static_dir_str
    except (ImportError, AttributeError):
        pass
    
    # Method 2: Use importlib_resources (Python 3.8)
    try:
        import importlib_resources
        static_dir = importlib_resources.files('mlx_audio').joinpath('tts')
        static_dir_str = str(static_dir)
        if os.path.exists(static_dir_str):
            return static_dir_str
    except ImportError:
        pass
    
    # Method 3: Use pkg_resources
    try:
        static_dir_str = pkg_resources.resource_filename('mlx_audio', 'tts')
        if os.path.exists(static_dir_str):
            return static_dir_str
    except (ImportError, pkg_resources.DistributionNotFound):
        pass
    
    # Method 4: Try to find the module path directly
    try:
        module_spec = importlib.util.find_spec('mlx_audio')
        if module_spec and module_spec.origin:
            package_dir = os.path.dirname(module_spec.origin)
            static_dir_str = os.path.join(package_dir, 'tts')
            if os.path.exists(static_dir_str):
                return static_dir_str
    except (ImportError, AttributeError):
        pass
    
    # Method 5: Look in sys.modules
    try:
        if 'mlx_audio' in sys.modules:
            module = sys.modules['mlx_audio']
            if hasattr(module, '__file__'):
                package_dir = os.path.dirname(module.__file__)
                static_dir_str = os.path.join(package_dir, 'tts')
                if os.path.exists(static_dir_str):
                    return static_dir_str
    except Exception:
        pass
    
    # If all methods fail, raise an error
    raise RuntimeError("Could not find static directory")


def get_fallback_html():
    """Return a fallback HTML page for the web interface."""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>MLX-Audio TTS Server</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                line-height: 1.6;
            }
            h1 {
                color: #333;
                border-bottom: 1px solid #ddd;
                padding-bottom: 10px;
            }
            .form-group {
                margin-bottom: 15px;
            }
            label {
                display: block;
                margin-bottom: 5px;
                font-weight: bold;
            }
            input[type="text"], select, textarea {
                width: 100%;
                padding: 8px;
                border: 1px solid #ddd;
                border-radius: 4px;
                box-sizing: border-box;
            }
            textarea {
                height: 100px;
                resize: vertical;
            }
            button {
                background-color: #4CAF50;
                color: white;
                padding: 10px 15px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
            }
            button:hover {
                background-color: #45a049;
            }
            .audio-container {
                margin-top: 20px;
                padding: 15px;
                background-color: #f9f9f9;
                border-radius: 4px;
            }
            .error {
                color: red;
                font-weight: bold;
            }
        </style>
    </head>
    <body>
        <h1>MLX-Audio Text-to-Speech</h1>
        
        <div class="form-group">
            <label for="text">Text to convert to speech:</label>
            <textarea id="text" placeholder="Enter text here..."></textarea>
        </div>
        
        <div class="form-group">
            <label for="voice">Voice:</label>
            <select id="voice">
                <option value="af_heart">AF Heart</option>
                <option value="af_nova">AF Nova</option>
                <option value="af_bella">AF Bella</option>
                <option value="bf_emma">BF Emma</option>
            </select>
        </div>
        
        <div class="form-group">
            <label for="speed">Speed: <span id="speed-value">1.0</span></label>
            <input type="range" id="speed" min="0.5" max="2.0" step="0.1" value="1.0">
        </div>
        
        <button id="generate">Generate Speech</button>
        
        <div id="error" class="error" style="display: none;"></div>
        
        <div id="audio-container" class="audio-container" style="display: none;">
            <h3>Generated Audio:</h3>
            <audio id="audio-player" controls></audio>
        </div>
        
        <script>
            document.getElementById('speed').addEventListener('input', function() {
                document.getElementById('speed-value').textContent = this.value;
            });
            
            document.getElementById('generate').addEventListener('click', function() {
                const text = document.getElementById('text').value;
                const voice = document.getElementById('voice').value;
                const speed = document.getElementById('speed').value;
                
                if (!text.trim()) {
                    showError('Please enter some text');
                    return;
                }
                
                // Hide previous error and audio
                document.getElementById('error').style.display = 'none';
                document.getElementById('audio-container').style.display = 'none';
                
                // Create form data
                const formData = new FormData();
                formData.append('text', text);
                formData.append('voice', voice);
                formData.append('speed', speed);
                
                // Send request to server
                fetch('/tts', {
                    method: 'POST',
                    body: formData
                })
                .then(response => {
                    if (!response.ok) {
                        return response.json().then(data => {
                            throw new Error(data.error || 'Failed to generate speech');
                        });
                    }
                    return response.json();
                })
                .then(data => {
                    // Set audio source and show player
                    const audioPlayer = document.getElementById('audio-player');
                    audioPlayer.src = `/audio/${data.filename}`;
                    document.getElementById('audio-container').style.display = 'block';
                    audioPlayer.play();
                })
                .catch(error => {
                    showError(error.message);
                });
            });
            
            function showError(message) {
                const errorElement = document.getElementById('error');
                errorElement.textContent = message;
                errorElement.style.display = 'block';
            }
        </script>
    </body>
    </html>
    """


def setup_server():
    """Setup the server by loading the model and creating the output directory."""
    global tts_model, OUTPUT_FOLDER
    
    # Make sure the output folder for generated TTS files exists
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Load the model if not already loaded
    if tts_model is None:
        tts_model = load_model(MODEL_PATH)
    
    # Try to mount the static files directory
    try:
        static_dir = find_static_dir()
        app.mount("/static", StaticFiles(directory=static_dir), name="static")
    except Exception as e:
        print(f"Warning: Could not mount static files directory: {e}")
        print("The server will still function, but the web interface may be limited.")


def main(host="127.0.0.1", port=8000):
    """Parse command line arguments for the server and start it."""
    parser = argparse.ArgumentParser(description="Start the MLX-Audio TTS server")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="Host address to bind the server to (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port to bind the server to (default: 8000)")
    args = parser.parse_args()
    
    # Start the server with the parsed arguments
    setup_server()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()