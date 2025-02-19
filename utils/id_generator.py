import uuid
from datetime import datetime

def generate_candidate_id() -> str:
    """Generate unique candidate ID"""
    return f"CAND_{datetime.now().strftime('%Y%m%d')}_{str(uuid.uuid4())[:8]}"

def generate_test_id() -> str:
    """Generate unique test ID"""
    return f"TEST_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:6]}" 