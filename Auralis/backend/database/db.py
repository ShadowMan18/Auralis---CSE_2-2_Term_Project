from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from contextlib import contextmanager
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.environ.get('DATABASE_URL')

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping = True,
)

SessionLocal = sessionmaker(
    bind = engine,
    autoflush = False,
    expire_on_commit = False
)

class Base(DeclarativeBase):
    pass 

@contextmanager
def get_db_connection():
    session = SessionLocal()

    try:
        yield session
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()