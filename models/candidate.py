from pydantic import BaseModel, EmailStr
from enum import Enum
from typing import Optional

class Domain(str, Enum):
    BUSINESS = "business"
    MARKETING = "marketing"
    SALES = "sales"

class SkillLevel(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    EXPERT = "expert"

class Candidate(BaseModel):
    id: str
    name: str
    email: EmailStr
    domain: Domain
    skill_level: SkillLevel
    resume_path: Optional[str] = None
    test_id: Optional[str] = None
    father_name: str
    photo_path: Optional[str] = None
    parsed_resume_data: Optional[dict] = None 