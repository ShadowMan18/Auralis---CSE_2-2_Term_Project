from flask import Flask, jsonify
from dotenv import load_dotenv
from flask_cors import CORS
from flask_jwt_extended import JWTManager
import logging
import os
import sys
from api_limiter import limiter

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    stream=sys.stdout
)

app = Flask(__name__)

CORS(app, resources={
    r'/*': {
        'origins': [f'http://localhost:{os.environ.get('FRONTEND_PORT') or 5000}'],   # change to deployed frontend url
        'methods': ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS'],
        'allow_headers': ['Content-Type', 'Authorization'],
        'expose_headers': ['Content-Type', 'Authorization'],
        'supports_credentials': True
    }
})

app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = int(os.environ.get('JWT_ACCESS_TOKEN_EXPIRES'))
app.config['JWT_REFRESH_TOKEN_EXPIRES'] = int(os.environ.get('JWT_REFRESH_TOKEN_EXPIRES'))
app.config['JWT_TOKEN_LOCATION'] = ['headers', 'cookies']
# app.config['JWT_COOKIE_SECURE'] = True          # add it only for deployment     
app.config['JWT_COOKIE_CSRF_PROTECT'] = True
jwt = JWTManager(app)

limiter.init_app(app)

@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({'error': 'Too many requests, please try again later'}), 429

#=== import blueprints here ===#
from auth_service.routes import auth
# from cloud_service.routes.s3 import cloud
# from cloud_service.routes.cloudinary import cloud       # choose the cloud provider (use aliasing if both needed)

#=== register blueprints here ===#
app.register_blueprint(auth.auth_bp)
# app.register_blueprint(cloud.cloud_bp)

@app.route('/')
def home():
    return jsonify({'message': 'This is backend'})

if __name__ == '__main__':
    app.run(
        host = '0.0.0.0', 
        port = os.environ.get('BACKEND_PORT'), 
        debug = os.environ.get('DEBUG', 'false').lower() == 'true'
    )