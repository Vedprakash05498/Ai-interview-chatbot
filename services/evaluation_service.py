# -*- coding: utf-8 -*-
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import spacy
from datetime import datetime
import json
import os

class EvaluationService:
    def __init__(self):
        # Download required NLTK data
        try:
            nltk.download('punkt')
            nltk.download('stopwords')
            self.nlp = spacy.load('en_core_web_sm')
        except Exception as e:
            print(f"Warning: NLP initialization error: {str(e)}")
            self.nlp = None
        
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 3))
        self.results_dir = "data/results"
        os.makedirs(self.results_dir, exist_ok=True)

    def evaluate_answer(self, candidate_answer: str, ideal_answer: str, keywords: list) -> dict:
        """Evaluate a single answer"""
        if not candidate_answer or not ideal_answer:
            return {
                'cosine_score': 0.0,
                'jaccard_score': 0.0,
                'keyword_score': 0.0,
                'total_score': 0.0
            }

        # Preprocess texts
        candidate_processed = self._preprocess_text(candidate_answer)
        ideal_processed = self._preprocess_text(ideal_answer)

        # Calculate Cosine Similarity
        try:
            tfidf_matrix = self.vectorizer.fit_transform([candidate_processed, ideal_processed])
            cosine_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        except:
            cosine_score = 0.0

        # Calculate Jaccard Similarity
        jaccard_score = self._calculate_jaccard_similarity(
            set(candidate_processed.split()),
            set(ideal_processed.split())
        )

        # Calculate Keyword Score
        keyword_score = self._calculate_keyword_score(candidate_processed, keywords)

        # Calculate Total Score (weighted average)
        total_score = (
            cosine_score * 0.4 +    # 40% weight to semantic similarity
            jaccard_score * 0.3 +   # 30% weight to content overlap
            keyword_score * 0.3      # 30% weight to keyword presence
        )

        return {
            'cosine_score': cosine_score,
            'jaccard_score': jaccard_score,
            'keyword_score': keyword_score,
            'total_score': total_score
        }

    def evaluate_interview(self, candidate_id: str, answers: dict, test_duration: int) -> dict:
        """Evaluate entire interview"""
        hr_scores = []
        domain_scores = []
        
        # Evaluate HR questions
        for q in answers.get('hr_questions', []):
            score = self.evaluate_answer(
                q.get('answer', ''),
                q.get('ideal_answer', ''),
                q.get('expected_keywords', [])
            )
            hr_scores.append(score['total_score'])

        # Evaluate domain questions
        for q in answers.get('domain_questions', []):
            score = self.evaluate_answer(
                q.get('answer', ''),
                q.get('ideal_answer', ''),
                q.get('expected_keywords', [])
            )
            domain_scores.append(score['total_score'])

        # Calculate statistics
        total_questions = len(hr_scores) + len(domain_scores)
        attempted_questions = sum(1 for score in hr_scores + domain_scores if score > 0)
        correct_answers = sum(1 for score in hr_scores + domain_scores if score >= 0.7)
        wrong_answers = attempted_questions - correct_answers

        # Calculate total score
        hr_total = sum(hr_scores) * (10/len(hr_scores))  # Scale to 10 points
        domain_total = sum(domain_scores) * (20/len(domain_scores))  # Scale to 20 points
        total_score = hr_total + domain_total
        percentage = (total_score / 30) * 100

        return {
            'candidate_id': candidate_id,
            'test_duration': test_duration,
            'questions_attempted': attempted_questions,
            'total_questions': total_questions,
            'correct_answers': correct_answers,
            'wrong_answers': wrong_answers,
            'hr_score': hr_total,
            'domain_score': domain_total,
            'total_score': total_score,
            'percentage': percentage
        }

    def save_results(self, results: dict, candidate_info: dict):
        """Save evaluation results to CSV"""
        df = pd.DataFrame([{
            'Serial_Number': self._get_next_serial_number(),
            'Candidate_ID': results['candidate_id'],
            'Candidate_Name': candidate_info.get('name', ''),
            'Email': candidate_info.get('email', ''),
            'Test_ID': f"TEST_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'Duration_Minutes': results['test_duration'] // 60,
            'Questions_Attempted': results['questions_attempted'],
            'Correct_Answers': results['correct_answers'],
            'Wrong_Answers': results['wrong_answers'],
            'HR_Score': results['hr_score'],
            'Domain_Score': results['domain_score'],
            'Total_Score': results['total_score'],
            'Percentage': results['percentage'],
            'Timestamp': datetime.now().isoformat()
        }])

        results_file = os.path.join(self.results_dir, 'evaluation_results.csv')
        if os.path.exists(results_file):
            df.to_csv(results_file, mode='a', header=False, index=False)
        else:
            df.to_csv(results_file, index=False)

    def save_feedback(self, candidate_id: str, feedback: dict):
        """Save candidate feedback"""
        feedback_file = os.path.join(self.results_dir, 'candidate_feedback.json')
        
        feedback_data = {
            'candidate_id': candidate_id,
            'timestamp': datetime.now().isoformat(),
            'feedback': feedback
        }

        existing_feedback = []
        if os.path.exists(feedback_file):
            with open(feedback_file, 'r') as f:
                existing_feedback = json.load(f)

        existing_feedback.append(feedback_data)

        with open(feedback_file, 'w') as f:
            json.dump(existing_feedback, f, indent=2)

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for evaluation"""
        # Convert to lowercase
        text = text.lower()
        
        # Tokenize and remove stopwords if spaCy is available
        if self.nlp:
            doc = self.nlp(text)
            tokens = [token.text for token in doc if not token.is_stop]
            return ' '.join(tokens)
        
        # Fallback to basic preprocessing
        stop_words = set(stopwords.words('english'))
        words = text.split()
        return ' '.join(w for w in words if w not in stop_words)

    def _calculate_jaccard_similarity(self, set1: set, set2: set) -> float:
        """Calculate Jaccard similarity between two sets"""
        if not set1 or not set2:
            return 0.0
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0

    def _calculate_keyword_score(self, text: str, keywords: list) -> float:
        """Calculate keyword presence score"""
        if not keywords:
            return 0.0
        text_words = set(text.split())
        keywords_found = sum(1 for kw in keywords if kw.lower() in text_words)
        return keywords_found / len(keywords)

    def _get_next_serial_number(self) -> int:
        """Get next serial number for results"""
        results_file = os.path.join(self.results_dir, 'evaluation_results.csv')
        if not os.path.exists(results_file):
            return 1
        df = pd.read_csv(results_file)
        return len(df) + 1 