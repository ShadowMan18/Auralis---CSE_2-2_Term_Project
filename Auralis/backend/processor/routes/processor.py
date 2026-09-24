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
    # latitude/longitude/month are optional as a group. A supplied region
    # should be resolved to coordinates by the client before this request.
    result, status = processor.process_cnn_sample(
        sample,
        lat=request.form.get('lat', request.form.get('latitude')),
        lon=request.form.get('lon', request.form.get('longitude')),
        month=request.form.get('month'),
        top_k=request.form.get('top_k', 8),
    )
    
    return jsonify(result), status
