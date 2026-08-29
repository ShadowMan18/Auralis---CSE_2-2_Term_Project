from flask import Blueprint, jsonify, request
import logging
import sys
from flask_jwt_extended import jwt_required
from demo_service.services import demo  # change import

logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s [%(levelname)s] %(message)s',
    stream=sys.stdout
)

demo_bp = Blueprint('demo', __name__)   # change bp name and update in app.py

@demo_bp.get('/api/demo/health')        # change endpoint
def health():
    return jsonify('alive!!!')

# === Add route handlers here === #

### demo_function_route ###
@demo_bp.post('/api/demo')
@jwt_required()
def demo_function_route():
    name = request.form.get('name')
    result, status = demo.demo_function(name)
    
    return jsonify(result), status