from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_jwt_extended import get_jwt_identity

def get_user_or_ip():
    try:
        return get_jwt_identity() or get_remote_address()
    except Exception:
        return get_remote_address()

limiter = Limiter(
    get_user_or_ip, 
    default_limits=["200 per day", "50 per hour"]
)