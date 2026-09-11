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

### process_sample ###
def process_sample(sample):
    print('received sample')

    return {'species': ['tiger', 'lion'], 'confidence': [0.9, 0.8]}, 201