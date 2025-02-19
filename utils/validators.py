import re
from typing import List
import magic

def validate_email(email: str) -> bool:
    """Validate email format"""
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return bool(re.match(pattern, email))

def validate_file_type(file_content: bytes, allowed_types: List[str]) -> bool:
    """Validate file MIME type"""
    mime = magic.Magic(mime=True)
    file_type = mime.from_buffer(file_content)
    return file_type in allowed_types

def validate_image(file_content: bytes) -> bool:
    """Validate if file is an image"""
    allowed_types = ['image/jpeg', 'image/png', 'image/jpg']
    return validate_file_type(file_content, allowed_types)

def validate_resume(file_content: bytes) -> bool:
    """Validate if file is a valid resume format"""
    allowed_types = [
        'application/pdf',
        'application/msword',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    ]
    return validate_file_type(file_content, allowed_types)

def validate_name(name: str) -> bool:
    """Validate name format"""
    return bool(re.match(r'^[A-Za-z\s]{2,50}$', name)) 