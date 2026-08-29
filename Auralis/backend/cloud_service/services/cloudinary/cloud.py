import logging
import sys
import os
import requests
import cloudinary
import cloudinary.uploader
import cloudinary.api
import cloudinary.utils
from dotenv import load_dotenv
import time

load_dotenv()

logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout
)

CLOUDINARY_CLOUD_NAME = os.environ.get('CLOUDINARY_CLOUD_NAME')
CLOUDINARY_API_KEY = os.environ.get('CLOUDINARY_API_KEY')
CLOUDINARY_API_SECRET = os.environ.get('CLOUDINARY_API_SECRET')

if not all([CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, CLOUDINARY_API_SECRET]):
    logging.error("Missing storage configuration in environment variables!")
    raise ValueError("Storage configuration incomplete. Check .env file.")

cloudinary.config(
    cloud_name = CLOUDINARY_CLOUD_NAME,
    api_key = CLOUDINARY_API_KEY,
    api_secret = CLOUDINARY_API_SECRET,
    secure = True
)

logging.info(f"Cloud client initialized for cloud: {CLOUDINARY_CLOUD_NAME}")


### check_connection ###
def check_connection():
    """
    Checks the cloud connection.

    Returns:
        bool: True if connected, False otherwise
    """

    try:
        cloudinary.api.ping()
        logging.info(f"Storage connection OK: {CLOUDINARY_CLOUD_NAME}")
        return True
    except Exception as e:
        logging.error(f"Storage connection failed: {e}")
        return False


### upload_file ###
def upload_file(file, key, resource_type = 'image', metadata = None):
    """
    Uploads file to the cloud.

    Args:
        file (str): File object
        key (str): Public ID (path in cloud), e.g., "music/song"
        resource_type (str): 'image', 'video' (also used for audio), or 'raw'
        metadata (dict): Optional metadata dict
    
    Returns:
        bool: True if successful, False otherwise
    
    Example:
        upload_file(
            "/tmp/song.mp3",
            "music/uuid123",
            "video",
            {"original-filename": "song.mp3"}
        )
    """

    try:
        extra_args = {'public_id': key, 'resource_type': resource_type}

        if metadata:
            extra_args['context'] = metadata

        cloudinary.uploader.upload(file, **extra_args)

        logging.info(f"Uploaded: {key} to {CLOUDINARY_CLOUD_NAME}")
        return True
    except FileNotFoundError as e:
        logging.error(f"File not found: {file}")
        return False
    except Exception as e:
        logging.error(f"Storage upload error: {e}")
        return False


### generate_presigned_url ###
def generate_presigned_url(key, resource_type = 'image', expires_in = 3600):
    """
    Generates presigned url for viewing private files.

    Args:
        key (str): Public ID (path in cloud), e.g., "music/song"
        resource_type (str): 'image', 'video' (also used for audio), or 'raw'
        expires_in (int): URL expiry time in seconds (default: 3600 = 1 hour)
    
    Returns:
        str: Signed URL or None if error
    
    Example:
        url = generate_presigned_url("music/uuid123", "video", expires_in = 3600)
        # Returns: "https://res.cloudinary.com/...?_a=...&Expires=..."
    """

    try:
        signed_url, _ = cloudinary.utils.cloudinary_url(
            key,
            resource_type = resource_type,
            type = 'authenticated',
            sign_url = True,
            auth_token = {'duration': expires_in}
        )

        logging.info(f"Generated signed URL for: {key} (expires in {expires_in}s)")
        return signed_url
    except Exception as e:
        logging.error(f"Error generating signed URL for {key}: {e}")
        return None


### generate_upload_presigned_url ###
def generate_upload_presigned_url(key, resource_type = 'image'):
    """
    Generates presigned params for uploading from frontend.
 
    Args:
        key (str): Public ID where file will be uploaded
        resource_type (str): 'image', 'video' (also used for audio), or 'raw'
    
    Returns:
        dict: {'signature', 'timestamp', 'api_key', 'cloud_name', 'resource_type', 'public_id'} or None if error
    """
 
    try:
        timestamp = int(time.time())
        params_to_sign = {'public_id': key, 'timestamp': timestamp}
        signature = cloudinary.utils.api_sign_request(params_to_sign, CLOUDINARY_API_SECRET)
 
        presigned_post = {
            'signature': signature,
            'timestamp': timestamp,
            'api_key': CLOUDINARY_API_KEY,
            'cloud_name': CLOUDINARY_CLOUD_NAME,
            'resource_type': resource_type,
            'public_id': key
        }
 
        logging.info(f"Generated upload presigned params for: {key}")
        return presigned_post
    except Exception as e:
        logging.error(f"Error generating upload URL: {e}")
        return None


### delete_file ###
def delete_file(key, resource_type = 'image'):
    """
    Deletes file from cloud.

    Args:
        key (str): Public ID to delete
        resource_type (str): 'image', 'video' (also used for audio), or 'raw'
    
    Returns:
        bool: True if successful, False otherwise
    
    Example:
        delete_file("music/uuid123", "video")
    """

    try:
        cloudinary.uploader.destroy(key, resource_type = resource_type)

        logging.info(f"Deleted: {key} from {CLOUDINARY_CLOUD_NAME}")
        return True
    except Exception as e:
        logging.error(f"Storage delete error for {key}: {e}")
        return False


### file_exists ###
def file_exists(key, resource_type = 'image'):
    """
    Checks if a file exists in the cloud.

    Args:
        key (str): Public ID to check
        resource_type (str): 'image', 'video' (also used for audio), or 'raw'
    
    Returns:
        bool: True if exists, False otherwise
    """

    try:
        cloudinary.api.resource(key, resource_type = resource_type)
        return True
    except cloudinary.exceptions.NotFound:
        return False
    except Exception as e:
        logging.error(f"Error checking file existence: {e}")
        return False


### list_files ###
def list_files(prefix = '', max_keys = 1000, resource_type = 'image'):
    """
    Gives the list of files in the cloud.

    Args:
        prefix (str): Prefix to filter (e.g., "music/" to list all music files)
        max_keys (int): Maximum number of files to return
        resource_type (str): 'image', 'video' (also used for audio), or 'raw'
    
    Returns:
        list: List of file keys, or empty list if error
    
    Example:
        files = list_files(prefix="music/", resource_type="video")
        # Returns: ["music/song1", "music/song2", ...]
    """

    try:
        response = cloudinary.api.resources(
            type = 'upload',
            prefix = prefix,
            max_results = max_keys,
            resource_type = resource_type
        )

        if 'resources' not in response:
            return []

        files = [obj['public_id'] for obj in response['resources']]

        logging.info(f"Listed {len(files)} files with prefix '{prefix}'")
        return files
    except Exception as e:
        logging.error(f"Error listing files: {e}")
        return []


### get_file_metadata ###
def get_file_metadata(key, resource_type = 'image'):
    """
    Gives the metadata of a file.

    Args:
        key (str): Public ID
        resource_type (str): 'image', 'video' (also used for audio), or 'raw'
    
    Returns:
        dict: File metadata (size, content_type, etc.) or None if error
    """

    try:
        response = cloudinary.api.resource(key, resource_type = resource_type)

        metadata = {
            'size': response.get('bytes'),
            'content_type': response.get('format'),
            'last_modified': response.get('created_at'),
            'metadata': response.get('context', {})
        }

        return metadata
    except Exception as e:
        logging.error(f"Error getting file metadata for {key}: {e}")
        return None


### download_file ###
def download_file(key, local_path, resource_type = 'image'):
    """
    Downloads a file from cloud to local storage.

    Args:
        key (str): Public ID to download
        local_path (str): Local path to save file
        resource_type (str): 'image', 'video' (also used for audio), or 'raw'
    
    Returns:
        bool: True if successful, False otherwise
    """

    try:
        url, _ = cloudinary.utils.cloudinary_url(key, resource_type = resource_type)
        response = requests.get(url)
        response.raise_for_status()

        with open(local_path, 'wb') as f:
            f.write(response.content)

        logging.info(f"Downloaded: {key} to {local_path}")
        return True
    except Exception as e:
        logging.error(f"Download error: {e}")
        return False