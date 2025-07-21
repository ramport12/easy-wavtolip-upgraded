import torch
import subprocess
import json
import os
import dlib
import gdown
import pickle
import re
from typing import Tuple, Optional, List, Any
import logging
from pathlib import Path
from models import Wav2Lip
from base64 import b64encode
from urllib.parse import urlparse
from torch.hub import download_url_to_file, get_dir
from IPython.display import HTML, display

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
logger.info(f"Using device: {device}")


def get_video_details(filename: str) -> Tuple[int, int, float, float]:
    """Get video details using ffprobe.
    
    Args:
        filename: Path to the video file
        
    Returns:
        Tuple of (width, height, fps, duration)
        
    Raises:
        subprocess.CalledProcessError: If ffprobe fails
        ValueError: If video stream not found or invalid format
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Video file not found: {filename}")
        
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_format",
        "-show_streams",
        "-of",
        "json",
        filename,
    ]
    
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        info = json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        logger.error(f"ffprobe failed: {e.stderr.decode()}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse ffprobe output: {e}")
        raise ValueError(f"Invalid ffprobe output for {filename}")

    # Get video stream
    video_streams = [stream for stream in info["streams"] if stream["codec_type"] == "video"]
    if not video_streams:
        raise ValueError(f"No video stream found in {filename}")
    
    video_stream = video_streams[0]

    # Get resolution
    width = int(video_stream["width"])
    height = int(video_stream["height"])

    # Get fps - safer evaluation
    fps_str = video_stream["avg_frame_rate"]
    if '/' in fps_str:
        num, den = map(float, fps_str.split('/'))
        fps = num / den if den != 0 else 25.0
    else:
        fps = float(fps_str)

    # Get length
    length = float(info["format"]["duration"])

    return width, height, fps, length


def show_video(file_path: str) -> None:
    """Function to display video in Colab.
    
    Args:
        file_path: Path to the video file
    """
    if not os.path.exists(file_path):
        logger.error(f"Video file not found: {file_path}")
        return
        
    try:
        with open(file_path, "rb") as f:
            mp4 = f.read()
        data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
        width, _, _, _ = get_video_details(file_path)
        display(
            HTML(
                """
      <video controls width=%d>
          <source src="%s" type="video/mp4">
      </video>
      """
                % (min(width, 1280), data_url)
            )
        )
    except Exception as e:
        logger.error(f"Failed to display video: {e}")


def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"


def _load(checkpoint_path):
    if device != "cpu":
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(
            checkpoint_path, map_location=lambda storage, loc: storage
        )
    return checkpoint


def load_model(path: str) -> torch.nn.Module:
    """Load Wav2Lip model from checkpoint with caching.
    
    Args:
        path: Path to the model checkpoint
        
    Returns:
        Loaded and evaluated model
        
    Raises:
        FileNotFoundError: If checkpoint file not found
        RuntimeError: If model loading fails
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
        
    # If cached results file exists, load it and return
    folder, filename_with_extension = os.path.split(path)
    filename, _ = os.path.splitext(filename_with_extension)
    results_file = os.path.join(folder, filename + ".pk1")
    
    if os.path.exists(results_file):
        try:
            logger.info(f"Loading cached model from {results_file}")
            with open(results_file, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cached model: {e}. Loading from checkpoint.")
            # Remove corrupted cache file
            os.remove(results_file)
    
    try:
        model = Wav2Lip()
        logger.info(f"Loading model from {path}")
        checkpoint = _load(path)
        
        if "state_dict" not in checkpoint:
            raise RuntimeError(f"Invalid checkpoint format: missing 'state_dict' in {path}")
            
        s = checkpoint["state_dict"]
        new_s = {}
        for k, v in s.items():
            new_s[k.replace("module.", "")] = v
        model.load_state_dict(new_s)

        model = model.to(device)
        model_eval = model.eval()
        
        # Save cached results to file
        try:
            with open(results_file, "wb") as f:
                pickle.dump(model_eval, f)
            logger.info(f"Model cached to {results_file}")
        except Exception as e:
            logger.warning(f"Failed to cache model: {e}")
            
        return model_eval
        
    except Exception as e:
        logger.error(f"Failed to load model from {path}: {e}")
        raise RuntimeError(f"Model loading failed: {e}")


def get_input_length(filename: str) -> float:
    """Get media file duration in seconds.
    
    Args:
        filename: Path to media file
        
    Returns:
        Duration in seconds
        
    Raises:
        subprocess.CalledProcessError: If ffprobe fails
        ValueError: If duration cannot be parsed
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Media file not found: {filename}")
        
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                filename,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        return float(result.stdout.decode().strip())
    except subprocess.CalledProcessError as e:
        logger.error(f"ffprobe failed for {filename}: {e.stderr.decode()}")
        raise
    except ValueError as e:
        logger.error(f"Failed to parse duration for {filename}: {e}")
        raise ValueError(f"Invalid duration format for {filename}")


def is_url(string):
    url_regex = re.compile(r"^(https?|ftp)://[^\s/$.?#].[^\s]*$")
    return bool(url_regex.match(string))


def load_predictor():
    checkpoint = os.path.join(
        "checkpoints", "shape_predictor_68_face_landmarks_GTX.dat"
    )
    predictor = dlib.shape_predictor(checkpoint)
    mouth_detector = dlib.get_frontal_face_detector()

    # Serialize the variables
    with open(os.path.join("checkpoints", "predictor.pkl"), "wb") as f:
        pickle.dump(predictor, f)

    with open(os.path.join("checkpoints", "mouth_detector.pkl"), "wb") as f:
        pickle.dump(mouth_detector, f)

    # delete the .dat file as it is no longer needed
    # os.remove(output)


def load_file_from_url(url, model_dir=None, progress=True, file_name=None):
    """Load file form http url, will download models if necessary.

    Ref:https://github.com/1adrianb/face-alignment/blob/master/face_alignment/utils.py

    Args:
        url (str): URL to be downloaded.
        model_dir (str): The path to save the downloaded model. Should be a full path. If None, use pytorch hub_dir.
            Default: None.
        progress (bool): Whether to show the download progress. Default: True.
        file_name (str): The downloaded file name. If None, use the file name in the url. Default: None.

    Returns:
        str: The path to the downloaded file.
    """
    if model_dir is None:  # use the pytorch hub_dir
        hub_dir = get_dir()
        model_dir = os.path.join(hub_dir, "checkpoints")

    os.makedirs(model_dir, exist_ok=True)

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if file_name is not None:
        filename = file_name
    cached_file = os.path.abspath(os.path.join(model_dir, filename))
    if not os.path.exists(cached_file):
        print(f'Downloading: "{url}" to {cached_file}\n')
        download_url_to_file(url, cached_file, hash_prefix=None, progress=progress)
    return cached_file


def g_colab():
    try:
        import google.colab

        return True
    except ImportError:
        return False


def get_mouth_landmarks_mediapipe(image: 'np.ndarray') -> Optional['np.ndarray']:
    """Extract mouth landmarks using MediaPipe.
    
    Args:
        image: Input image as numpy array (BGR format)
        
    Returns:
        Array of mouth landmark points or None if no face detected
    """
    try:
        import mediapipe as mp
        import cv2
        import numpy as np
    except ImportError as e:
        logger.error(f"Required package not available: {e}")
        return None
        
    if image is None or len(image.shape) != 3:
        logger.warning("Invalid input image for mouth landmark detection")
        return None
        
    try:
        mp_face_mesh = mp.solutions.face_mesh
        with mp_face_mesh.FaceMesh(
            static_image_mode=True, 
            max_num_faces=1, 
            refine_landmarks=True,
            min_detection_confidence=0.5
        ) as face_mesh:
            # Convert image to RGB
            if image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_image)
            
            if not results.multi_face_landmarks:
                return None
                
            landmarks = results.multi_face_landmarks[0].landmark
            # Mouth landmark indices for MediaPipe Face Mesh
            mouth_indices = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 
                           324, 318, 402, 317, 14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 
                           267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]
            
            h, w, _ = image.shape
            mouth_points = np.array([[int(landmarks[i].x * w), int(landmarks[i].y * h)] 
                                   for i in mouth_indices])
            return mouth_points
            
    except Exception as e:
        logger.error(f"MediaPipe mouth landmark detection failed: {e}")
        return None
