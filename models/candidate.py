from pydantic import BaseModel, EmailStr
from enum import Enum
from typing import Optional

class Domain(str, Enum):
    AI_ML = "AI/ML"
    WEB_DEV = "Web Development"
    BUSINESS = "Business Analyst"
    MARKETING = "Marketing"
    SALES = "Sales"

class SkillLevel(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    EXPERT = "expert"

class Candidate(BaseModel):
    candidate_id: str
    test_id: str
    name: str
    father_name: str
    email: EmailStr
    domain: Domain
    resume_path: Optional[str] = None
    photo_path: Optional[str] = None
    camera_enabled: bool = False
    mobile_camera_connected: bool = False
    parsed_resume_data: Optional[dict] = None
    warnings_count: int = 0 