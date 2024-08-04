# AcademAI

**AcademAI** is a FastAPI-based application that generates and evaluates educational questions and answers for students. The app uses LLM model to assess student responses on conceptual understanding, problem-solving, and clarity of expression.

## Features

- **Generate Questions:** Generate customized questions based on topic, difficulty level, and subject.
- **Evaluate Answers:** Assess student answers using a machine learning model to provide scores and detailed feedback.
- **Final Report:** Generate a final report summarizing the student's performance, including overall percentage scores and topic-wise marks.

## Endpoints

### 1. Root

- **GET /**  
  Returns a welcome message.

### 2. Generate Questions

- **GET /api/v1/academai/questions**  
  Generates questions based on the specified topic, subject, difficulty level, and number of questions.

  **Parameters:**
  - `student_id`: int
  - `difficulty_level`: str
  - `questions`: int (default = 1)
  - `topic_name`: str
  - `subject`: str

### 3. Evaluate Answers

- **POST /api/v1/academai/evaluate**  
  Evaluates student answers and returns a detailed breakdown of scores and feedback.

  **Request Body:**
  - `student_id`: int
  - `subject`: str
  - `topic`: str
  - `answers`: List of `question_id` and `answer`

### 4. Final Report

- **GET /api/v1/academai/final_report**  
  Returns a summary report including feedback and scores based on evaluated answers.

  **Parameters:**
  - `student_id`: int
  - `subject`: str

## Setup & Installation
   ```bash
   1. git clone https://github.com/yourusername/academai.git
   2. cd academai
   3. pip install -r requirements.txt
   4. uvicorn main:app --reload
   5. Access the API at http://localhost:8000.


Or
Run the below commoand to run the app

source init_setup.sh
