import sys
import os
import io

# Add the backend directory to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.quiz_generator import generate_quiz
from app.pdf_processor import extract_text_from_pdf

def test_quiz_generator():
    dummy_text = (
        "Photosynthesis is a process used by plants to convert light energy into chemical energy. "
        "Mitochondria are double-membraned organelles found in most eukaryotic organisms. "
        "They generate most of the cell's supply of adenosine triphosphate. "
        "Gravity is a fundamental interaction that causes mutual attraction between all things with mass. "
        "Python is an interpreted high-level general-purpose programming language. "
        "Flutter is a UI software development kit created by Google. "
        "It is used to develop cross-platform applications."
    )
    print("=== Testing Quiz Generator ===")
    questions = generate_quiz(dummy_text)
    print(f"Generated {len(questions)} questions.\n")
    
    for i, q in enumerate(questions):
        print(f"Question {i+1}: {q.question_text}")
        print(f"Options: {q.options}")
        print(f"Correct Option: Index {q.correct_option_index} ({q.options[q.correct_option_index]})")
        print(f"Explanation: {q.explanation}")
        print("-" * 40)
        
    assert len(questions) > 0, "Error: No questions generated."
    print("Quiz generator test: PASSED\n")

if __name__ == "__main__":
    test_quiz_generator()
