from datetime import datetime, timedelta

class Timer:
    def __init__(self, duration_seconds: int):
        self.duration = duration_seconds
        self.start_time = None
        self.end_time = None
        
    def start(self):
        """Start the timer"""
        self.start_time = datetime.now()
        self.end_time = self.start_time + timedelta(seconds=self.duration)
        
    def time_remaining(self) -> int:
        """Get remaining time in seconds"""
        if not self.start_time:
            return self.duration
        
        remaining = (self.end_time - datetime.now()).total_seconds()
        return max(0, int(remaining))
    
    def is_expired(self) -> bool:
        """Check if timer has expired"""
        return self.time_remaining() <= 0 