import pandas as pd
from sklearn.model_selection import KFold, cross_validate
import yaml
import os
import numpy as np
import json
import joblib

# загружаем данные из sprint 1: update_data.csv
if os.path.exists('/home/mle-user/mle_projects/mle-project-sprint-1-v001/part2_dvc/data/update_data.csv'):
    data = pd.read_csv('/home/mle-user/mle_projects/mle-project-sprint-1-v001/part2_dvc/data/update_data.csv')

# сохраняем данные
os.makedirs('../data', exist_ok=True)
with open('../data/initial_data.csv', 'wb') as fd:
    data.to_csv('../data/initial_data.csv', index=None)
    
# загружаем модель из sprint 1: fitted_model_CBR.pkl
if os.path.exists('/home/mle-user/mle_projects/mle-project-sprint-1-v001/part2_dvc/models/fitted_model_CBR.pkl'):
    with open('/home/mle-user/mle_projects/mle-project-sprint-1-v001/part2_dvc/models/fitted_model_CBR.pkl', 'rb') as fd:
        model = joblib.load(fd)

# сохраняем модель
os.makedirs('../models', exist_ok=True) # создание директории, если её ещё нет
with open('../models/base_model_CBR.pkl', 'wb') as fd:
    joblib.dump(model, fd)
    
