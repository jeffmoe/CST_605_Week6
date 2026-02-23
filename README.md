# CST_605_Week6  
Repo for the week 6 assignment.  
Running the application successfully:  
$ export KAGGLE_API_TOKEN=xxxxxxxxxxxxxx  
$ export ENABLE_FILE_LOGS=1  
$ export LOG_LEVEL=DEBUG  
$ export LOG_DIR=Logs  
go to the repo and activate the venv:  $ source venv/bin/activate  
make sure the venv has all the modules needed from the requirements.txt file.  
Run the model trainer file: $ python model_train.py  
Load the app: $ uvicorn app:app --host 0.0.0.0 --port 8000 --reload  

The feature EDA files are in the data processed folder and the reports folder.  
The weather_api.py file is for the virtual lab and not apart of the final file submission.  
