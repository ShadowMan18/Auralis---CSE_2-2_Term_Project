from flask import Blueprint, jsonify, request
import logging
import sys
from flask_jwt_extended import jwt_required
from cloud_service.services.cloudinary import cloud
from cloud_service.utils import ALLOWED_TYPES, get_file_extension
import uuid
import time

logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s [%(levelname)s] %(message)s',
    stream=sys.stdout
)

cloud_bp = Blueprint('cloud', __name__)   

@cloud_bp.get('/api/cloud/health')        
def health():
    return jsonify('alive!!!')

### generate_upload_presigned_url_route ###
@cloud_bp.post('/api/cloud/presigned-upload')
@jwt_required()
def generate_upload_presigned_url_route():
    data = request.get_json(silent = True) or {}
    filename = data.get('filename')
    content_type = data.get('content_type')
    category = data.get('category')
 
    if not all([filename, content_type, category]):
        return jsonify({'error': 'filename, content_type, and category are required'}), 400
 
    rules = ALLOWED_TYPES.get(category)
    if rules is None:
        return jsonify({'error': f'Unknown category: {category}'}), 400
 
    ext = get_file_extension(filename)
    if rules['extensions'] and (ext is None or ext not in rules['extensions']):
        return jsonify({'error': f'File type not allowed for {category}'}), 400
    if rules['mime_types'] and content_type not in rules['mime_types']:
        return jsonify({'error': f'File type not allowed for {category}'}), 400
 
    key = f"{category}s/{uuid.uuid4()}" + (f".{ext}" if ext else "")
    presigned = cloud.generate_upload_presigned_url(key, resource_type=rules['resource_type'])
 
    if presigned is None:
        return jsonify({'error': 'Could not generate upload credentials'}), 500
 
    return jsonify(presigned), 200
 
 
### confirm_upload_route ###
@cloud_bp.post('/api/cloud/confirm-upload')
@jwt_required()
def confirm_upload_route():
    data = request.get_json(silent = True) or {}
    key = data.get('key')
    category = data.get('category')
 
    if not key or not category:
        return jsonify({'error': 'Key and category are required'}), 400
 
    rules = ALLOWED_TYPES.get(category)
    if rules is None:
        return jsonify({'error': f'Unknown category: {category}'}), 400
 
    confirmed = False
    for _ in range(3):          # change try count
        if cloud.file_exists(key, resource_type=rules['resource_type']):
            confirmed = True
            break
        time.sleep(1)
 
    if not confirmed:
        return jsonify({'error': 'Upload could not be verified. Please try again.'}), 400

    metadata = cloud.get_file_metadata(key, resource_type = rules['resource_type'])
    if metadata is None:
        return jsonify({'error': 'Could not verify upload'}), 500

    if metadata['size'] > rules['max_size']:
        cloud.delete_file(key, resource_type = rules['resource_type'])
        max_mb = rules['max_size'] // (1024 * 1024)
        return jsonify({'error': f'File exceeds maximum size of {max_mb}MB'}), 400
 
    return jsonify({'message': 'Upload successful'}), 200
 