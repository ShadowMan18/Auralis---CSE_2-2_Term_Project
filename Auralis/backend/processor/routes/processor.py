from flask import Blueprint, jsonify, request
import logging
import sys
from flask_jwt_extended import jwt_required
from processor.services import processor  

logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s [%(levelname)s] %(message)s',
    stream=sys.stdout
)

processor_bp = Blueprint('processor', __name__)   

@processor_bp.get('/api/processor/health')        
def health():
    return jsonify('alive!!!')

### process_sample_route ###
@processor_bp.post('/api/upload-sample')
@jwt_required()
def process_sample_route():
    sample = request.files.get('sample')
    result, status = processor.process_sample(sample)
    
    return jsonify(result), status