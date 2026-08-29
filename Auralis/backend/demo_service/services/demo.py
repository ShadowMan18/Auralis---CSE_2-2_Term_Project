import logging
import sys
from flask_jwt_extended import get_jwt_identity
from database.db import get_db_connection
from sqlalchemy import *
from database.models import Demo

logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s [%(levelname)s] %(message)s',
    stream=sys.stdout
)

#=== Add function implementations here ===#

### demo_function ###
def demo_function(name):
    with get_db_connection() as db:
        item = Demo(name = name)
        db.add(item)
        
    return {'message': 'success'}, 201