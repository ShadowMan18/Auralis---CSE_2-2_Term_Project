from flask import jsonify
import logging
import sys
from flask_jwt_extended import create_access_token, create_refresh_token, set_refresh_cookies
from database.db import get_db_connection
from sqlalchemy import *
from sqlalchemy.exc import IntegrityError
from database.models import User
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import date

logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s [%(levelname)s] %(message)s',
    stream=sys.stdout
)

### signup ###
def signup(user_type, name, email, password, date_of_birth):
    user = User(
        user_type = user_type,
        name = name,
        email = email,
        password = generate_password_hash(password),
        date_of_birth = date.fromisoformat(str(date_of_birth)),
        profile_picture = None
    )

    try:
        with get_db_connection() as db:
            db.add(user)
    except IntegrityError:
        return {'error': 'A user with this email already exists'}, 400

    with get_db_connection() as db:
        user_id = db.scalars(
            select(User).where(User.user_type == user_type, User.email == email)
        ).first().user_id

    token = create_access_token(identity = str(user_id))

    return (
        {
            'user_id': user_id,
            'token': token
        }
    ), 201


### onboarding ###
def onboarding(user_id, profile_picture):
    with get_db_connection() as db:
        user = db.get(User, user_id)
        if user is None:
            return {'error': 'User not found'}, 404
        user.profile_picture = profile_picture

    return {'message': 'successful'}, 200


### login ###
def login(user_type, email, password):
    with get_db_connection() as db:
        result = db.scalars(
            select(User).where(User.user_type == user_type, User.email == email)
        ).first()

        if result is None:
            return {'error': 'User not found'}, 404

        if not check_password_hash(result.password, password):
            return {'error': 'Invalid password'}, 400

        access_token = create_access_token(identity = str(result.user_id))
        refresh_token = create_refresh_token(identity = str(result.user_id))

        response = {
            'user_id': result.user_id,
            'name': result.name,
            'profile_picture': result.profile_picture,
            'token': access_token
        }

        set_refresh_cookies(jsonify(response), refresh_token)

        return response, 200