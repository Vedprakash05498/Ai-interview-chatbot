# AI Interview Chatbot

An intelligent interview system that analyzes resumes and conducts domain-specific technical interviews.

## Features

- **Resume Analysis**
  - PDF, DOC, DOCX support
  - Skills extraction
  - Experience analysis
  - Education details parsing
  - Domain detection (Web Development/AI-ML)
  - Skill level assessment

- **Dynamic Question Generation**
  - Domain-specific questions (Web Dev/AI-ML)
  - 15 questions per interview
    - 5 Easy questions
    - 5 Medium questions
    - 5 Hard questions
  - Adaptive difficulty based on skill level

- **Evaluation System**
  - Real-time answer evaluation
  - Score calculation (out of 100)
  - Detailed feedback
  - Strengths and weaknesses analysis
  - Improvement recommendations

## Project Structure

```
ai-interview-chatbot/
├── api/
│   ├── routes/
│   │   ├── candidate.py
│   │   ├── interview.py
│   │   ├── proctor.py
│   │   └── evaluation.py
├── services/
│   ├── resume_service.py
│   ├── question_service.py
│   ├── proctor_service.py
│   ├── evaluation_service.py
│   └── camera_service.py
├── models/
│   ├── candidate.py
│   ├── interview.py
│   ├── proctor.py
│   ├── feedback.py
│   └── evaluation.py
├── data/
│   ├── questions/
│   │   ├── ai_ml/
│   │   ├── web_dev/
│   │   ├── business/
│   │   ├── marketing/
│   │   ├── sales/
│   │   └── hr/
│   ├── responses/
│   ├── uploads/
│   │   ├── resumes/
│   │   ├── photos/
│   │   └── snapshots/
│   └── results/
├── static/
│   ├── js/
│   └── css/
├── templates/
└── utils/
    ├── id_generator.py
    ├── time_utils.py
    └── validators.py
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/VedprakashRAD/Ai-interview-chatbot.git
cd Ai-interview-chatbot
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Configuration

Update `config.py` with your settings:
```python
# config.py
import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-here'
    UPLOAD_FOLDER = 'uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx'}
```

## Running the Application

1. Start the server:
```bash
python app.py
```

2. Access the API at `http://127.0.0.1:8004`

## API Endpoints

### 1. Resume Upload
```http
POST /api/resume/upload
Content-Type: multipart/form-data

file: resume.pdf/doc/docx
```

Response:
```json
{
    "skills": ["Python", "FastAPI", "Machine Learning"],
    "experience": ["Senior Developer", "Tech Lead"],
    "education": ["Master in Computer Science"],
    "domain": "ai_ml",
    "skill_level": "intermediate"
}
```

### 2. Get Questions
```http
GET /api/questions/{domain}/{skill_level}
```

Response:
```json
[
    {
        "id": "q1",
        "category": "ai_ml",
        "difficulty": "intermediate",
        "question": "Explain neural networks",
        "expected_keywords": ["layers", "neurons", "activation"]
    }
]
```

### 3. Submit Evaluation
```http
POST /api/evaluation/evaluate
Content-Type: application/json

{
    "answers": [
        {
            "question_id": "q1",
            "answer": "Detailed answer here"
        }
    ]
}
```

Response:
```json
{
    "score": 85,
    "feedback": "Excellent understanding of concepts",
    "strengths": ["Technical knowledge", "Clear explanation"],
    "weaknesses": ["Could add more examples"],
    "recommendations": ["Practice more coding questions"]
}
```

## Directory Details

### API Structure
- `api/routes/`: Contains all API endpoint definitions
- `api/models/`: Data models and schemas
- `services/`: Business logic implementation
- `data/`: Question banks for different domains

### Question Categories
- **AI/ML**:
  - AI fundamentals
  - Machine Learning concepts
  - Advanced topics
- **Web Development**:
  - HTML/CSS
  - JavaScript
  - Frontend/Backend concepts

## Development

1. Install development dependencies:
```bash
pip install pytest black flake8
```

2. Run tests:
```bash
python -m pytest
```

3. Format code:
```bash
black .
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

# This project is licensed under the MIT License.

## Contact

Vedprakash - [@VedprakashRAD](https://github.com/VedprakashRAD)

Project Link: [https://github.com/VedprakashRAD/Ai-interview-chatbot](https://github.com/VedprakashRAD/Ai-interview-chatbot)
```
