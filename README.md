# **Manual MLOps Pipeline – Predictive Maintenance SystemManual MLOps Pipeline – Predictive Maintenance System**

# 

### Project Overview



This project implements a manual end-to-end MLOps pipeline for predictive maintenance using a machine failure dataset.



The objective is to predict machine failure based on operational and sensor measurements, while demonstrating:



* Manual data versioning
* Configuration isolation
* Feature engineering
* Model training \& versioning
* API-based deployment
* Production logging
* Monitoring \& drift simulation
* Automated retraining trigger



This repository simulates a real-world production lifecycle without using external MLOps platforms.

The model uses the following features:



* Type
* Air temperature \[K]
* Process temperature \[K]
* Rotational speed \[rpm]
* Torque \[Nm]
* Tool wear \[min]



Manual-MLOps-project/

│

├── config.yaml

├── manifest.txt

│

├── data/

│   ├── raw/

│   ├── processed/

│

├── models/

│

├── logs/

│   ├── deployment\_log.csv

│   ├── production\_predictions.jsonl

│

├── src/

│   ├── data\_prep.py

│   ├── train.py

│   ├── inference.py

│   ├── monitor\_production.py

│

├── tests/

│

└── requirements.txt



#### CLONE REPOSITORY



git clone https://github.com/LavanyaDiviti891/Manual-MLOps-project.git

cd Manual-MLOps-project



#### INSTALL DEPENDENCIES



pip install -r requirements.txt



Phase - 1 DATA PREPARATION \& VERSIONING

Raw dataset is stored inside:

data/raw/



##### Run:

python src/data\_prep.py



##### This step:



* Cleans the dataset
* Applies feature engineering
* Encodes categorical variables
* Scales numerical features



Saves processed dataset to:

data/processed/



Updates manifest.txt to maintain manual data lineage



This ensures traceability from raw → processed versions.



### PHASE - 2: MODEL TRAINING



Train the model using:

python src/train.py



All hyperparameters and file paths are controlled through:

config.yaml



Example parameters:

* n\_estimators
* max\_depth
* train/test split
* model save path



No hardcoding is used in the training script.



##### The trained model is saved inside:



models/



Model versioning is maintained manually.

### 

### PHASE - 3: API DEPLOYMENT



#### Start the FastAPI server:



uvicorn src.inference:app --reload



API Documentation available at:



http://127.0.0.1:8000/docs



#### STEP - 2 Run Smoke Test

Open a new Terminal:



python src/smoke\_tests.py



Purpose of Smoke Test:

* Validates API connectivity
* Confirms correct response structure
* Ensures prediction endpoint is functioning





#### PREDICTION ENDPOINT



Sample Input:



{

&nbsp; "features": \["L", 298.1, 308.6, 1551, 42.8, 0]

}



#### The API:



* Loads model path from config.yaml
* Returns prediction



Logs each prediction to:



logs/production\_predictions.jsonl



### PHASE - 4: DEPLOYMENT LOGGING 

### 

##### Deployment history is maintained in:



logs/deployment\_log.csv



This log tracks:

* Timestamp
* Model version
* Performance metrics
* Activation history



This ensures traceability of which model was active at a given time.



### PHASE - 5: MONITORING \& DRIFT SIMULATION



#### Production Evaluation – Day 2 Simulation

##### 

##### To simulate production drift and evaluate model performance:



python src/run\_day2\_inference.py



#### To simulate production monitoring:



python src/monitor\_production.py



This step:



1. Reads production predictions
2. Calculates production accuracy
3. Compares against threshold from config.yaml
4. Logs monitoring results
5. Automatically triggers retraining if performance drops below threshold



#### RETRAIN LOGIC



Threshold is defined inside:

monitoring:

&nbsp; accuracy\_threshold: 0.80



If production accuracy < threshold → retraining is triggered.



The instructor can modify the threshold in config.yaml without changing any code















































