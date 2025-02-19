from pydantic import BaseModel
from datetime import datetime
from typing import List, Dict, Optional

class InterviewSession(BaseModel):
    candidate_id: str
    test_id: str
    start_time: datetime
    end_time: Optional[datetime]
    hr_questions: List[Dict]
    domain_questions: List[Dict]
    current_question_index: int = 0
    hr_time_remaining: int = 600  # 10 minutes in seconds
    domain_time_remaining: int = 1200  # 20 minutes in seconds
    is_completed: bool = False 