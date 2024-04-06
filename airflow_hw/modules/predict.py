# <YOUR_IMPORTS>
import datetime
import json
import os

import dill

import pandas as pd


def predict():
    path = os.environ.get('PROJECT_PATH', '.') # Путь к файлу
    mod = sorted(os.listdir(f'{path}/data/models'))
    with open(f'{path}/data/models/cars_pipe_202404042340.pkl', 'rb') as file:
        model = dill.load(file)

    df_pred = pd.DataFrame(columns=['id', 'pred'])
    files_list = os.listdir(f'{path}/data/test')

    for filename in files_list:
        with open(f'{path}/data/test/{filename}') as file:
            form = json.load(file)
        data = pd.DataFrame.from_dict([form])
        prediction = model.predict(data)
        dict_pred = {'id': data.id, 'pred': prediction}
        df = pd.DataFrame(dict_pred)
        df_pred = pd.concat([df, df_pred], axis=0)
        df = df_pred
    df.to_csv(f'{path}/data/predictions/{datetime.datetime.now().strftime("%Y%m%d%H%M")}.csv',
              index=False)


if __name__ == '__main__':
    predict()
