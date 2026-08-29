from database.db import Base
from sqlalchemy.orm import Mapped, mapped_column

# define the model

class Demo(Base):
    __tablename__ = 'demo'

    id: Mapped[int] = mapped_column(primary_key = True)
    name: Mapped[str]

# add import to __init__.py