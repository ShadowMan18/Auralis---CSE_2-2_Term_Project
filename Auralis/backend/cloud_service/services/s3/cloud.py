import logging
import sys
import os
import boto3
from botocore.client import Config
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout
)

CLOUD_ENDPOINT = os.environ.get('S3_CLOUD_ENDPOINT')
CLOUD_ACCESS_KEY = os.environ.get('S3_CLOUD_ACCESS_KEY')
CLOUD_SECRET_KEY = os.environ.get('S3_CLOUD_SECRET_KEY')
CLOUD_BUCKET = os.environ.get('S3_CLOUD_BUCKET')
CLOUD_REGION = os.environ.get('S3_CLOUD_REGION')

if not all([CLOUD_ENDPOINT, CLOUD_ACCESS_KEY, CLOUD_SECRET_KEY, CLOUD_BUCKET, CLOUD_REGION]):
    logging.error("Missing storage configuration in environment variables!")
    raise ValueError("Storage configuration incomplete. Check .env file.")

s3_client = boto3.client(
    's3',
    endpoint_url = CLOUD_ENDPOINT,
    aws_access_key_id = CLOUD_ACCESS_KEY,
    aws_secret_access_key = CLOUD_SECRET_KEY,
    config = Config(signature_version = 's3v4'),
    region_name = CLOUD_REGION
)

logging.info(f"Cloud client initialized for bucket: {CLOUD_BUCKET}")


### check_connection ###
def check_connection():
    """
    Checks the cloud connection.

    Returns:
        bool: True if connected, False otherwise
    """

    try:
        s3_client.head_bucket(Bucket = CLOUD_BUCKET)
        logging.info(f"Storage connection OK: {CLOUD_BUCKET}")
        return True
    except ClientError as e:
        logging.error(f"Storage connection failed: {e}")
        return False
    

### upload_file ###
def upload_file(file, key, content_type, metadata = None):
    """
    Uploads file to the cloud.

    Args:
        file (str): File object
        key (str): S3 key (path in bucket), e.g., "music/song.mp3"
        content_type (str): MIME type, e.g., "audio/mpeg"
        metadata (dict): Optional metadata dict
    
    Returns:
        bool: True if successful, False otherwise
    
    Example:
        upload_file(
            "/tmp/song.mp3",
            "music/uuid123.mp3",
            "audio/mpeg",
            {"original-filename": "song.mp3"}
        )
    """
    
    try:
        extra_args = {'ContentType': content_type}
        
        if metadata:
            extra_args['Metadata'] = metadata
        
        s3_client.upload_fileobj(
            Fileobj = file,
            Bucket = CLOUD_BUCKET,
            Key = key,
            ExtraArgs = extra_args
        )
        
        logging.info(f"Uploaded: {key} to {CLOUD_BUCKET}")
        return True
    except ClientError as e:
        logging.error(f"Storage upload error: {e}")
        return False
    except FileNotFoundError as e:
        logging.error(f"File not found: {file}")
        return False
    except Exception as e:
        logging.error(f"Unexpected upload error: {e}")
        return False


### generate_presigned_url ###
def generate_presigned_url(key, expires_in = 3600):
    """
    Generates presigned url for viewing private files.

    Args:
        key (str): S3 key (path in bucket), e.g., "music/song.mp3"
        expires_in (int): URL expiry time in seconds (default: 3600 = 1 hour)
    
    Returns:
        str: Signed URL or None if error
    
    Example:
        url = generate_signed_url("music/uuid123.mp3", expires_in = 3600)
        # Returns: "https://...backblazeb2.com/...?X-Amz-Signature=..."
    """

    try:
        signed_url = s3_client.generate_presigned_url(
            'get_object',
            Params = {
                'Bucket': CLOUD_BUCKET,
                'Key': key
            },
            ExpiresIn = expires_in
        )
        
        logging.info(f"Generated signed URL for: {key} (expires in {expires_in}s)")
        return signed_url
    except ClientError as e:
        logging.error(f"Error generating signed URL for {key}: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error generating signed URL: {e}")
        return None


### generate_upload_presigned_url ###
def generate_upload_presigned_url(key, content_type, expires_in = 900):
    """
    Generates presigned url for uploading from frontend.
    
    """
    try:
        url = s3_client.generate_presigned_url(
            'put_object',
            Params={
                'Bucket': CLOUD_BUCKET,
                'Key': key,
                'ContentType': content_type,
            },
            ExpiresIn=expires_in,
        )
        logging.info(f"Generated upload presigned URL for: {key}")
        return {'url': url}
    except ClientError as e:
        logging.error(f"Error generating upload URL: {e}")
        return None
    

### delete_file ###
def delete_file(key):
    """
    Deletes file from cloud.

    Args:
        key (str): S3 key (path in bucket) to delete
    
    Returns:
        bool: True if successful, False otherwise
    
    Example:
        delete_file("music/uuid123.mp3")
    """

    try:
        s3_client.delete_object(
            Bucket = CLOUD_BUCKET,
            Key = key
        )
        
        logging.info(f"Deleted: {key} from {CLOUD_BUCKET}")
        return True
    except ClientError as e:
        logging.error(f"Storage delete error for {key}: {e}")
        return False
    except Exception as e:
        logging.error(f"Unexpected delete error: {e}")
        return False


### file_exists ###
def file_exists(key):
    """
    Checks if a file exists in the cloud.

    Args:
        key (str): S3 key to check
    
    Returns:
        bool: True if exists, False otherwise
    """

    try:
        s3_client.head_object(Bucket = CLOUD_BUCKET, Key = key)
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            return False
        logging.error(f"Error checking file existence: {e}")
        return False


### list_files ###
def list_files(prefix = '', max_keys = 1000):
    """
    Gives the list of files in the cloud.

    Args:
        prefix (str): Prefix to filter (e.g., "music/" to list all music files)
        max_keys (int): Maximum number of files to return
    
    Returns:
        list: List of file keys, or empty list if error
    
    Example:
        files = list_files(prefix="music/")
        # Returns: ["music/song1.mp3", "music/song2.mp3", ...]
    """

    try:
        response = s3_client.list_objects_v2(
            Bucket = CLOUD_BUCKET,
            Prefix = prefix,
            MaxKeys = max_keys
        )
        
        if 'Contents' not in response:
            return []
        
        files = [obj['Key'] for obj in response['Contents']]

        logging.info(f"Listed {len(files)} files with prefix '{prefix}'")
        return files
    except ClientError as e:
        logging.error(f"Error listing files: {e}")
        return []


### get_file_metadata ###
def get_file_metadata(key):
    """
    Gives the metadata of a file.

    Args:
        key (str): S3 key
    
    Returns:
        dict: File metadata (size, content_type, etc.) or None if error
    """

    try:
        response = s3_client.head_object(Bucket = CLOUD_BUCKET, Key = key)
        
        metadata = {
            'size': response.get('ContentLength'),
            'content_type': response.get('ContentType'),
            'last_modified': response.get('LastModified'),
            'metadata': response.get('Metadata', {})
        }
        
        return metadata
    except ClientError as e:
        logging.error(f"Error getting file metadata for {key}: {e}")
        return None


### download_file ###
def download_file(key, local_path):
    """
    Downloads a file from file to local storage.

    Args:
        key (str): S3 key to download
        local_path (str): Local path to save file
    
    Returns:
        bool: True if successful, False otherwise
    """

    try:
        s3_client.download_file(CLOUD_BUCKET, key, local_path)
        logging.info(f"Downloaded: {key} to {local_path}")
        return True
    except ClientError as e:
        logging.error(f"Download error: {e}")
        return False