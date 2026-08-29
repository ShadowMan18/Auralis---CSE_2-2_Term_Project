from database.db import Base
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import UniqueConstraint
from datetime import date

class User(Base):
    __tablename__ = 'user'
    __table_args__ = (
        UniqueConstraint('user_type', 'email', name = 'uq_user_type_email'),
    )

    user_id: Mapped[int] = mapped_column(primary_key = True)
    user_type: Mapped[str] = mapped_column(nullable = False)
    name: Mapped[str] = mapped_column(nullable = False)
    email: Mapped[str] = mapped_column(nullable = False)
    password: Mapped[str] = mapped_column(nullable = False)
    date_of_birth: Mapped[date] = mapped_column(nullable = False)
    profile_picture: Mapped[str] = mapped_column(nullable = True)