from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional

class ProctorAlert(BaseModel):
    timestamp: datetime
    alert_type: str  # multiple_faces, no_face, suspicious_movement, etc.
    screenshot_path: Optional[str]
    warning_count: int

class ProctorSession(BaseModel):
    candidate_id: str
    test_id: str
    start_time: datetime
    alerts: List[ProctorAlert] = []
    is_mobile_camera_active: bool = False
    is_webcam_active: bool = False 