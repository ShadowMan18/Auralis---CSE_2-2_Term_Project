from flask import Blueprint, jsonify, request
import logging
import sys
from flask_jwt_extended import jwt_required, get_jwt_identity, create_access_token, unset_jwt_cookies
from auth_service.services import auth
from auth_service.validation_schema import SignupSchema, LoginSchema
from pydantic import ValidationError
from api_limiter import limiter

logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s [%(levelname)s] %(message)s',
    stream=sys.stdout
)

auth_bp = Blueprint('auth', __name__)

@auth_bp.get('/api/auth/health')
def health():
    return jsonify('alive!!!')

### signup_route ###
@auth_bp.post('/api/auth/signup')
@limiter.limit('5 per hour')
def signup_route():
    try:
        user = SignupSchema(**request.form.to_dict())
    except ValidationError as e:
        errors = {err['loc'][0]: err['msg'] for err in e.errors()}
        return jsonify({'error': 'Validation failed', 'details': errors}), 400
    
    result, status = auth.signup(user.user_type, user.name, user.email, user.password, user.date_of_birth)

    return jsonify(result), status


### onboarding_route ###
@auth_bp.post('/api/auth/onboarding')
@jwt_required()
def onboarding_route():
    user = request.form
    user_id = int(user.get('user_id'))
    profile_picture = user.get('profile_picture') or None
    result, status = auth.onboarding(user_id, profile_picture)

    return jsonify(result), status


### login_route ###
@auth_bp.post('/api/auth/login')
@limiter.limit('10 per minute')
def login_route():
    try:
        user = LoginSchema(**request.form.to_dict())
    except ValidationError as e:
        errors = {err['loc'][0]: err['msg'] for err in e.errors()}
        return jsonify({'error': 'Validation failed', 'details': errors}), 400

    result, status = auth.login(user.user_type, user.email, user.password)

    return jsonify(result), status


### logout_route ###
@auth_bp.post('/api/auth/logout')
def logout_route():
    response = jsonify({'message': 'logged out'})
    unset_jwt_cookies(response)

    return response, 200


### refresh_route ###
@auth_bp.post('/api/auth/refresh')
def refresh_route():
    identity = get_jwt_identity()
    access_token = create_access_token(identity = identity)
    
    return jsonify({'token': access_token}), 200