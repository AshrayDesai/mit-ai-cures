# mit-ai-cures
MIT AI Cures is a project wherein many data scientists attempt to use machine learning to solve medical problems such as the COVID-19 pandemic. Here is my experimentation with machine learning models on data from the AI Cures database. Currently, I have trained and tested an XGBoost gradient-boosting machine learning model on a dataset of SMILES molecule representations to predict whether or not new molecules will be able to bind to and disable sars-cov-2, the novel coronavirus behind the disease COVID-19.
- XGBoostOnMITAICuresData_workbook.v1.7.ipynb: My XGBoost model and training/testing/evaluation of the model on the dataset I chose.
- smileencodev2: script to encode SMILES string molecule representations into vectors processable by XGBoost
