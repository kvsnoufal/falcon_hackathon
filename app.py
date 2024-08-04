from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from ai71 import AI71  
import os
import json
from dotenv import load_dotenv
import uuid 
import pandas as pd
from datetime import datetime
from pathlib import Path
import csv
from typing import List

# Initialize FastAPI
app = FastAPI() 

# Load environment variables from .env file
load_dotenv('.env')

# API KEY
AI71_API_KEY = os.getenv('AI71_API_KEY')
client = AI71(AI71_API_KEY)

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


def calculate_marks_per_topic(filtered_df):
    # Group by 'topic' and sum the marks
    marks_per_topic = filtered_df.groupby('topic')['mark'].sum().reset_index()
    
    # Calculate the number of questions per topic
    num_questions_per_topic = filtered_df.groupby('topic').size().reset_index(name='num_questions')
    
    # Merge the number of questions with the marks per topic
    marks_per_topic = marks_per_topic.merge(num_questions_per_topic, on='topic')
    
    # Calculate the total possible marks for each topic
    marks_per_topic['total_possible_marks'] = marks_per_topic['num_questions'] * 10
    
    return marks_per_topic

def llm_to_get_feedback(summary):
    # Construct the prompt for the LLM
    prompt = (
        f"Based on the following suggestions summary, provide a JSON response with"
        f"improvements and strengths:\n\nSummary: {summary}\n\n"
        "Provide the response as a JSON object with keys strengths and gaps. Both the keys should contain points based on the provided summary to improve."
    )
    
    response = client.chat.completions.create(
        model="tiiuae/falcon-180B-chat", 
        messages=[
            {"role": "system", "content": "You are a teaching assistant."},
            {"role": "user", "content": prompt},
                ],
        temperature=0.9)
        
    # Extract the response
    feed_back = response.choices[0].message.content
    
    # Step 1: Clean the string by removing unwanted characters
    cleaned_text = feed_back.replace('\n', '').replace('} {', '},{').replace('}{', '},{')
        
    #print(cleaned_text)
    
    # Step 2: Convert cleaned text to a JSON list
    feed_back_json = json.loads(f"[{cleaned_text}]")
    
    # Assuming the model returns a valid JSON string, we can return it directly
    return feed_back_json

# Define a function to evaluate the answer
def evaluate_answer_using_model(question, given_answer, actual_answer):
    
    prompt = f"Evaluate the following answer for the question '{question}':\n\nAnswer: {given_answer}\n\n Correct Answer: {actual_answer}\n\n Provide a json response with mark which should be a value in the range of 10 and suggestions which contains the improvements to be done mainly."
    
    response = client.chat.completions.create(model="tiiuae/falcon-180B-chat", 
        messages=[
            {"role": "system", "content": "You are a teaching assistant and strict in evaluation when you provide the mark."},
            {"role": "user", "content": prompt},
                ],temperature=0.5)
    
    return response.choices[0].message.content
   
#Base
@app.get("/")
def read_root():
    return {"Welcome to AcademAI!"}

# Endpoint to generate questions
@app.get("/api/v1/academai/questions")
async def get_question(student_id: str,difficulty_level: str, questions: int = Query(1, gt=0), topic_name: str = Query(...), 
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

        response = client.chat.completions.create(
        model="tiiuae/falcon-180B-chat", 
        messages=[
            {"role": "system", "content": "You are a teaching assistant."},
            {"role": "user", "content": prompt},
                ],
        temperature=0.5)
        
        # Extract the generated questions
        generated_questions = response.choices[0].message.content
        
        # Step 1: Clean the string by removing unwanted characters
        cleaned_text = generated_questions.replace('\n', '').replace('} {', '},{').replace('}{', '},{')
        
        # Generate a unique ID for the file
        qa_folder='tmp/db.csv'
        unique_id = str(uuid.uuid4())
        
        # Step 2: Convert cleaned text to a JSON list
        questions_and_answers = json.loads(f"[{cleaned_text}]")
        
        # Save the questions and answers to a CSV file
        with open(qa_folder, 'a', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=['unique_id', 'question_id','topic','question', 'answer'])
            
            for qa in questions_and_answers:
                question_id = str(uuid.uuid4())
                writer.writerow({
                    'unique_id': unique_id,
                    'question_id':question_id,
                    'topic':topic_name,
                    'question': qa['question'],
                    'answer': qa['answer']
                })
        
        # Append metadata to a CSV file
        metadata_file_path = 'tmp/metadata.csv'
        metadata = {
            'student_id': student_id, 
            'unique_id': unique_id,
            'subject': subject,
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        metadata_df = pd.DataFrame([metadata])
        
        if os.path.exists(metadata_file_path):
            metadata_df.to_csv(metadata_file_path, mode='a', header=False, index=False)
        else:
            metadata_df.to_csv(metadata_file_path, mode='w', header=True, index=False)
            
        qa=pd.read_csv(qa_folder)
        qa=qa[qa['unique_id']==unique_id]
        #print(qa)
    
        # Prepare the response
        questions_only = qa[['question_id', 'question']].to_dict(orient='records')
        #print(questions_only)
        return {"questions": questions_only}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint to evaluate
@app.post("/api/v1/academai/evaluate")
async def evaluate_answer(request: EvaluateRequest):
    
    student_id = request.student_id
    subject = request.subject
    topic = request.topic
    answers = request.answers

    # Load CSV data into a DataFrame
    metadata_df = pd.read_csv('tmp/metadata.csv')

    # Get unique_id based on student_id and subject
    student_data = metadata_df[(metadata_df['student_id'] == student_id) & (metadata_df['subject'] == subject)]
    
    if student_data.empty:
        raise HTTPException(status_code=404, detail="Student or subject not found")
    
    unique_id = student_data.iloc[0]['unique_id']
    
    csv_file_path = Path(f"tmp/db.csv")
    if not csv_file_path.exists():
        raise HTTPException(status_code=404, detail="Q&A file not found")
    
    # Load the Q&A CSV file into a DataFrame
    qa_df = pd.read_csv(csv_file_path)
    
    evaluation_folder = 'tmp/evaluation.csv'
    
    # Prepare the list to collect evaluation results
    results = []
    
    for answer in answers:
        # Get the correct answer from the CSV file
        correct_answer_row = qa_df[(qa_df['unique_id'] == unique_id) & (qa_df['question_id'] == answer.question_id)]
        
        if correct_answer_row.empty:
            results.append({
                'question_id': answer.question_id,
                'mark': None,
                'suggestions': 'Question not found in Q&A file'
            })
            continue
        
        actual_answer = correct_answer_row['answer'].values[0]
        
        #qa_df[qa_df['question_id'] == answer.question_id]['question']

        # Evaluate the answer
        evaluation = evaluate_answer_using_model(qa_df[qa_df['question_id'] == answer.question_id]['question'], answer.answer, actual_answer)
        
        evaluation=json.loads(evaluation)
        
        # Append the evaluation results
        results.append({
            'question_id': answer.question_id,
            'mark': evaluation.get('mark', None),
            'suggestions': evaluation.get('suggestions', 'No suggestions available')
        })
        
        # Save the evaluation results to a CSV file
        with open(evaluation_folder, 'a', newline='') as csv_file:
            fieldnames = ['student_id', 'unique_id', 'subject', 'topic', 'question_id','mark', 'suggestions']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            
            # Write header only if the file is empty
            if csv_file.tell() == 0:
                writer.writeheader()
            
            writer.writerow({
                'student_id': student_id,
                'unique_id': unique_id,
                'subject': subject,
                'topic': topic,
                'question_id': answer.question_id,
                'mark': evaluation['mark'],
                'suggestions': evaluation['suggestions']
            })
    
    return {"results": results}

#Endpoint for performance report
@app.get("/api/v1/academai/final_report")
async def get_feedback(student_id: int, subject: str):
    
    meta_data_df = pd.read_csv('tmp/metadata.csv')
    evaluation_df = pd.read_csv('tmp/evaluation.csv')
    
    # Filter the meta_data by student_id and subject, and get the latest unique_id
    filtered_meta_data = meta_data_df[(meta_data_df['student_id'] == student_id) & 
                                      (meta_data_df['subject'] == subject)]
    
    if filtered_meta_data.empty:
        raise HTTPException(status_code=404, detail="No data found for the given student_id and subject.")
    
    latest_meta_data = filtered_meta_data.sort_values('created_at', ascending=False).iloc[0]
  
    unique_id = latest_meta_data['unique_id']
   
    
    # Get the suggestions from the evaluation dataset using the unique_id
    suggestions = evaluation_df[evaluation_df['unique_id'] == unique_id]['suggestions'].tolist()
    
    if not suggestions:
        raise HTTPException(status_code=404, detail="No suggestions found for the given unique_id.")
    
    # Concatenate the suggestions into a single summary
    summary = " ".join(suggestions)
    
    # Call LLM to get feedback (assumed to be an existing function)
    feedback_json = llm_to_get_feedback(summary)
    
    # Filter the DataFrame by unique_id and subject
    filtered_df = evaluation_df[(evaluation_df['unique_id'] == unique_id) & (evaluation_df['subject'] == subject)]
    
    # Calculate the sum of marks
    total_marks_obtained = filtered_df['mark'].sum()
    
    # Calculate the total possible marks (10 * number of rows)
    total_possible_marks = 10 * len(filtered_df)
    
    # Calculate the percentage score
    percentage_score = (total_marks_obtained / total_possible_marks) * 100

    # Calculate marks per topic
    marks_per_topic = calculate_marks_per_topic(filtered_df)
    
    response = {
        "feedback": feedback_json,
        "percentage_marks": percentage_score,
        "marks_per_topic": marks_per_topic.to_dict(orient='records')
    }
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
