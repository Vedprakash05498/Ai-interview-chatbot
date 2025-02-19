# -*- coding: utf-8 -*-
import json
import os
from pathlib import Path
import random
from typing import List, Dict
from datetime import datetime

class QuestionService:
    def __init__(self):
        self.datasets = {
            'business': 'data/business/business_analyst_questions.json',
            'hr': 'data/hr/hr_questions.json',
            'marketing': 'data/marketing/marketing_questions.json',
            'sales': 'data/Sales/sales_questions.json'
        }
        self.questions_cache = {}
        self._load_questions()
    
    def _load_questions(self):
        """Load all question datasets"""
        for domain, path in self.datasets.items():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    self.questions_cache[domain] = json.load(f)['questions']
            except Exception as e:
                print(f"Error loading {domain} questions: {str(e)}")
    
    def determine_domain(self, skills: List[str]) -> str:
        """Determine candidate's domain based on skills"""
        domain_keywords = {
            'business': {'business analysis', 'strategy', 'management', 'analytics', 'consulting'},
            'marketing': {'marketing', 'advertising', 'branding', 'social media', 'content'},
            'sales': {'sales', 'negotiation', 'client', 'account management', 'business development'}
        }

        domain_scores = {domain: 0 for domain in domain_keywords.keys()}
        
        # Convert skills to lowercase for matching
        skills_lower = [skill.lower() for skill in skills]
        
        # Calculate score for each domain
        for domain, keywords in domain_keywords.items():
            for skill in skills_lower:
                if any(keyword in skill for keyword in keywords):
                    domain_scores[domain] += 1

        # Get domain with highest score
        selected_domain = max(domain_scores.items(), key=lambda x: x[1])[0]
        return selected_domain if domain_scores[selected_domain] > 0 else 'business'  # Default to business

    def get_interview_questions(self, domain: str, skill_level: str) -> Dict:
        """Get HR and domain-specific questions for interview"""
        # Always get 5 HR questions
        hr_questions = self._get_random_questions('hr', 5)
        
        # Get 10 domain-specific questions based on skill level
        domain_questions = self._get_random_questions(
            domain, 
            10, 
            difficulty=skill_level
        )

        return {
            'hr_questions': hr_questions,
            'domain_questions': domain_questions,
            'time_limits': {
                'hr_questions': 600,  # 10 minutes in seconds
                'domain_questions': 1200  # 20 minutes in seconds
            }
        }

    def _get_random_questions(self, domain: str, count: int, difficulty: str = None) -> List[Dict]:
        """Get random questions from specified domain and difficulty"""
        if domain not in self.questions_cache:
            raise ValueError(f"Invalid domain: {domain}")

        questions = self.questions_cache[domain]
        
        # Filter by difficulty if specified
        if difficulty and domain != 'hr':
            questions = [q for q in questions if q.get('difficulty', '').lower() == difficulty.lower()]

        # If not enough questions available, use all questions
        count = min(count, len(questions))
        
        return random.sample(questions, count)

    def validate_answer(self, question: Dict, answer: str) -> float:
        """Validate answer against expected keywords"""
        if not answer or not question.get('expected_keywords'):
            return 0.0

        answer_lower = answer.lower()
        keywords_found = sum(1 for keyword in question['expected_keywords'] 
                           if keyword.lower() in answer_lower)
        
        return keywords_found / len(question['expected_keywords'])

    def store_answers(self, candidate_id: str, answers: Dict):
        """Store candidate's answers"""
        filename = f"data/responses/{candidate_id}_answers.json"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w') as f:
            json.dump({
                'candidate_id': candidate_id,
                'timestamp': datetime.now().isoformat(),
                'answers': answers
            }, f, indent=2)

    def get_questions_by_category(self, category):
        """Get questions for a specific category"""
        if category not in self.questions_cache:
            return []  # Return empty list instead of raising exception
        return self.questions_cache[category]
    
    def get_next_question(self, previous_question_id, answer):
        """Get next question based on previous answer"""
        if not previous_question_id:
            # If no previous question, return first question from any category
            for questions in self.questions_cache.values():
                if questions:
                    return questions[0]
            return None
            
        # Find next question in sequence
        for questions in self.questions_cache.values():
            for i, q in enumerate(questions):
                if q['id'] == previous_question_id and i + 1 < len(questions):
                    return questions[i + 1]
        return None

    def get_questions_for_interview(self, domain: str, skill_level: str) -> List[dict]:
        """Get 15 questions (5 easy, 5 medium, 5 hard)"""
        questions = []
        
        # Load questions based on domain
        if domain == 'web_dev':
            categories = ['html', 'css', 'javascript']
        else:  # ai_ml
            categories = ['ai', 'ml', 'advanced']

        for category in categories:
            category_questions = self.questions_cache.get(category, [])
            
            # Filter by difficulty
            easy = [q for q in category_questions if q['difficulty'] == 'beginner']
            medium = [q for q in category_questions if q['difficulty'] == 'intermediate']
            hard = [q for q in category_questions if q['difficulty'] == 'expert']
            
            # Select questions
            questions.extend(random.sample(easy, min(2, len(easy))))
            questions.extend(random.sample(medium, min(2, len(medium))))
            questions.extend(random.sample(hard, min(1, len(hard))))
        
        # Shuffle questions
        random.shuffle(questions)
        return questions[:15] 