echo "[ `date` ]": "START"
echo "[ `date` ]": "Creating virtual env" 
python -m venv venv/
echo "[ `date` ]": "activate venv"
source venv/bin/activate
echo "[ `date` ]": "installing the requirements" 
pip install -r requirements.txt
echo "[ `date` ]": "Running the Application" 
uvicorn app:app --reload
echo "[ `date` ]": "END"