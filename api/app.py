from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import os
import json
import uuid
import pandas as pd
from datetime import datetime
from typing import List
from services import llm_response, evaluate_answer_using_model, llm_to_get_feedback, calculate_marks_per_topic
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI
app = FastAPI() 

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])

# In-memory storage for Q&A, metadata, and evaluation result
qa_df = pd.DataFrame(columns=[
    'unique_id', 'question_id', 'topic', 'question', 'answer', 
    'conceptual_understanding', 'problem_solving', 'clarity_of_expression'])

metadata_df = pd.DataFrame(columns=['student_id', 'unique_id', 'subject', 'created_at'])
evaluation_df = pd.DataFrame(columns=['student_id', 'unique_id', 'subject', 'topic', 'question_id', 'mark', 'suggestions'])

class AnswerRequest(BaseModel):
    question_id: str
    question: str
    
class AnswerCheck(BaseModel):
    question_id: str
    answer: str
    
class EvaluateRequest(BaseModel):
    student_id: int
    subject: str
    topic: str
    answers: List[AnswerCheck]

@app.get("/")
def read_root():
    return {"Welcome to AcademAI!"}

@app.get("/api/v1/academai/questions")
async def get_question(student_id: int, difficulty_level: str, questions: int = Query(1, gt=0), topic_name: str = Query(...), 
                       subject: str = Query(...)):
    try:
        # Path to the data folder and topic file
        data_folder = f"data/{subject}"
        topic_file_path = os.path.join(data_folder, f"{topic_name}.txt")
        
        # Check if the file exists
        if not os.path.exists(topic_file_path):
            raise HTTPException(status_code=404, detail="Topic data not found")
        
        # Load the topic data from the text file
        with open(topic_file_path, 'r') as file:
            topic_data = file.read().strip()
        
        # Create the prompt for generating questions
        prompt = f"""
        Based on the following topic, generate {questions} questions with a difficulty level of {difficulty_level} for a 10th grade student 
        from the below chapter of a book.

        Topic: {topic_name}

        Chapter: {topic_data}

        The questions and answers should be in json format with keys 'question' and 'answer' with the values being the question and answer respectively. 
        """
        has_error = False
        for _ in range(3):
            # Extract the generated questions
            generated_questions = llm_response(prompt, temp=0.9)
            
            # Step 1: Clean the string by removing unwanted characters and ensure proper formatting
            cleaned_text = generated_questions.replace('\n', '').replace('} {', '},{').replace('}{', '},{')
            
            # Debugging: Print the cleaned text to check for issues
            print(f"Cleaned JSON text: {cleaned_text}")
            
            # Wrap the text with array brackets if not already done
            if not cleaned_text.startswith('['):
                cleaned_text = f"[{cleaned_text}]"
            
            try:
                # Step 2: Convert cleaned text to a JSON list
                questions_and_answers = json.loads(cleaned_text)
                has_error = False
            except json.JSONDecodeError as json_err:
                # Log the exact position of the error for debugging
                print(f"Error in JSON at line {json_err.lineno} column {json_err.colno}: {json_err.msg}")
                print("retrying attempt ",_+1)
                has_error = True

            if has_error == False:
                break
        if has_error:
            raise HTTPException(status_code=502, detail=f"JSON parsing error: {json_err.msg} at line {json_err.lineno} column {json_err.colno}")
            
        # Generate a unique ID for the file
        unique_id = str(uuid.uuid4())
        
        # Convert the list of questions and answers to a DataFrame
        global qa_df
        new_entries = pd.DataFrame([
            {
                'unique_id': unique_id,
                'question_id': str(uuid.uuid4()),
                'topic': topic_name,
                'question': qa['question'],
                'answer': qa['answer']
            } for qa in questions_and_answers
        ])
        
        # Concatenate new entries to the in-memory DataFrame
        qa_df = pd.concat([qa_df, new_entries], ignore_index=True)
        
        # Prepare metadata and concatenate to in-memory DataFrame
        global metadata_df
        metadata = {
            'student_id': student_id, 
            'unique_id': unique_id,
            'subject': subject,
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        metadata_df = pd.concat([metadata_df, pd.DataFrame([metadata])], ignore_index=True)
        
        # Prepare the response
        questions_only = new_entries[['question_id', 'question']].to_dict(orient='records')
        return {"questions": questions_only}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/academai/evaluate")
async def evaluate_answer(request: EvaluateRequest):
    
    student_id = request.student_id
    subject = request.subject
    topic = request.topic
    answers = request.answers

    global metadata_df
    global qa_df
    global evaluation_df

    # Get unique_id based on student_id and subject
    student_data = metadata_df[(metadata_df['student_id'] == student_id) & (metadata_df['subject'] == subject)]
    
    print(student_data)
    if student_data.empty:
        raise HTTPException(status_code=404, detail="Student or subject not found")
    
    unique_id = student_data.iloc[0]['unique_id']
    print(unique_id)
    # Prepare the list to collect evaluation results
    results = []
    
    for answer in answers:
        print(answer,answer.question_id)
        # Get the correct answer from the DataFrame
        correct_answer_row = qa_df[(qa_df['unique_id'] == unique_id) & (qa_df['question_id'] == answer.question_id)]
        
        if correct_answer_row.empty:
            results.append({
                'question_id': answer.question_id,
                'mark': None,
                'conceptual_understanding': None,
                'problem_solving': None,
                'clarity_of_expression': None,
                'suggestions': 'Question not found in Q&A data'
            })
            continue
        
        actual_answer = correct_answer_row['answer'].values[0]
        
        # Evaluate the answer
        evaluation = evaluate_answer_using_model(correct_answer_row['question'].values[0], answer.answer, actual_answer)
        evaluation = json.loads(evaluation)
        
        # Append the evaluation results
        results.append({
            'question_id': answer.question_id,
            'mark': evaluation.get('mark', None),
            'conceptual_understanding': evaluation.get('conceptual_understanding', None),
            'problem_solving': evaluation.get('problem_solving', None),
            'clarity_of_expression': evaluation.get('clarity_of_expression', None),
            'suggestions': evaluation.get('suggestions', 'No suggestions available')
        })
        
        # Save the evaluation results to in-memory DataFrame
        new_evaluations = pd.DataFrame([{
            'student_id': student_id,
            'unique_id': unique_id,
            'subject': subject,
            'topic': topic,
            'question_id': answer.question_id,
            'mark': evaluation['mark'],
            'conceptual_understanding': evaluation.get('conceptual_understanding', None),
            'problem_solving': evaluation.get('problem_solving', None),
            'clarity_of_expression': evaluation.get('clarity_of_expression', None),
            'suggestions': evaluation['suggestions']
        }])
        
        evaluation_df = pd.concat([evaluation_df, new_evaluations], ignore_index=True)
    
    return {"results": results}

@app.get("/api/v1/academai/final_report")
async def get_feedback(student_id: int, subject: str):
    
    global metadata_df
    global evaluation_df
    
    # Filter the meta_data by student_id and subject, and get the latest unique_id
    filtered_meta_data = metadata_df[(metadata_df['student_id'] == student_id) & 
                                      (metadata_df['subject'] == subject)]
    
    if filtered_meta_data.empty:
        raise HTTPException(status_code=404, detail="No data found for the given student_id and subject.")
    
    latest_meta_data = filtered_meta_data.sort_values('created_at', ascending=False).iloc[0]
  
    unique_id = latest_meta_data['unique_id']
   
    # Get the suggestions from the evaluation dataset using the unique_id
    suggestions = evaluation_df[evaluation_df['unique_id'] == unique_id]['suggestions'].tolist()
    
    if not suggestions:
        raise HTTPException(status_code=404, detail="No suggestions found for the given unique_id.")
    
    # Flatten the list and concatenate the suggestions into a single summary
    flat_suggestions = [item for sublist in suggestions for item in sublist]
    
    # Concatenate the suggestions into a single summary
    summary = " ".join(flat_suggestions)
    
    # Call LLM to get feedback
    feedback_json = llm_to_get_feedback(summary)
    
    # Filter the DataFrame by unique_id and subject
    filtered_df = evaluation_df[(evaluation_df['unique_id'] == unique_id) & (evaluation_df['subject'] == subject)]
    
    # Calculate the sum of marks and other metrics
    total_marks_obtained = filtered_df['mark'].sum()
    total_conceptual_understanding = filtered_df['conceptual_understanding'].sum()
    total_problem_solving = filtered_df['problem_solving'].sum()
    total_clarity_of_expression = filtered_df['clarity_of_expression'].sum()
    
    num_questions = len(filtered_df)
    
    # Calculate the total possible marks (10 * number of rows)
    total_possible_marks = 10 * num_questions
    total_possible_conceptual_understanding = 10 * num_questions
    total_possible_problem_solving = 10 * num_questions
    total_possible_clarity_of_expression = 10 * num_questions
    
    # Calculate the percentage score
    percentage_score = (total_marks_obtained / total_possible_marks) * 100
    conceptual_understanding_score = (total_conceptual_understanding / total_possible_conceptual_understanding) * 100
    problem_solving_score = (total_problem_solving / total_possible_problem_solving) * 100
    clarity_of_expression_score = (total_clarity_of_expression / total_possible_clarity_of_expression) * 100

    # Calculate marks per topic
    marks_per_topic = calculate_marks_per_topic(filtered_df)
    
    response = {
        "feedback": feedback_json,
        "percentage_marks": percentage_score,
        "conceptual_understanding": conceptual_understanding_score,
        "problem_solving": problem_solving_score,
        "clarity_of_expression": clarity_of_expression_score,
        "marks_per_topic": marks_per_topic.to_dict(orient='records')
    }
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
