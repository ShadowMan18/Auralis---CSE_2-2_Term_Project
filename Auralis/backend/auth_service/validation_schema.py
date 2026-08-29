from pydantic import BaseModel, EmailStr, field_validator
from typing import Literal
from datetime import date
import re

class SignupSchema(BaseModel):
    user_type: Literal['admin', 'client']
    name: str
    email: EmailStr
    password: str
    date_of_birth: date

    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        v = v.strip()
        if len(v) < 1:
            raise ValueError('Please enter name')
        if len(v) > 30:
            raise ValueError('Name is too long')
        return v

    @field_validator('password')
    @classmethod
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must contain at least 8 characters')
        if not re.search(r'[a-z]', v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not re.search(r'[0-9]', v):
            raise ValueError('Password must contain at least one number')
        if not re.search(r'[!@#$%&]', v):
            raise ValueError('Password must contain at least one special character')
        return v


class LoginSchema(BaseModel):
    user_type: Literal['admin', 'client']
    email: EmailStr
    password: str