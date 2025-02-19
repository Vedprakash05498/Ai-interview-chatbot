from pydantic import BaseModel
from typing import Optional

class Feedback(BaseModel):
    experience_rating: int
    interface_rating: int
    difficulty_rating: int
    clarity_rating: int
    comments: Optional[str] = None
    suggestions: Optional[str] = None 