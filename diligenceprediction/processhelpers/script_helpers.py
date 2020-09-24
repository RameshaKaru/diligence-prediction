# imports dependencies
import pandas as pd
import numpy as np
import yaml
import datetime
from dateutil.relativedelta import relativedelta
from skfuzzy.cluster import cmeans, cmeans_predict
from tensorflow import keras as keras
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import classification_report


from .mergedates import MergeDates
from .campdatehelpers import arrangeDate, afterSystemOnline


def read_data(fileName):
    """
      Reads Excel file

      Returns
      -------
      df : pandas DataFrame
          Dataframe with unique mothers
      dfr : pandas DataFrame
          Dataframe (original)
      """

    dfr = pd.read_excel(fileName)
    df = dfr.drop_duplicates(subset=['ANC_Mother Id'], keep='first', inplace=False).reset_index()
    return df, dfr


def get_pred_sub_centers(configs):
    next_sub_center_ids_file = configs['next_sub_center_ids_file']
    df = pd.read_csv(next_sub_center_ids_file)
    pred_sub_centers = np.array(df['sub_center_id'])
    return pred_sub_centers


def get_saved_model():
    lstm_model_cm = keras.models.load_model('models/regression_model')
    return lstm_model_cm


def get_cluster_centers():
    df = pd.read_csv('fixedinputs/cmeans_centers.csv')
    center1 = np.array(df["cluster1"])
    center2 = np.array(df["cluster2"])
    cntr = np.append(center1.reshape(center1.shape[0], 1), center2.reshape(center2.shape[0], 1), axis=1)
    return cntr.T.shape


def get_configs():
    with open('config.yaml', 'r') as stream:
        try:
            configs = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    return configs


def get_sliding_dates(online_day, data_end_day, window_size_weeks=4):
    online_day = online_day + relativedelta(days=-1)
    slide_start_day = online_day + relativedelta(weeks=+window_size_weeks)

    sliding_dates = [[window_size_weeks]]
    window_dates = []
    last_day = slide_start_day

    while last_day <= data_end_day:
        window_dates.append(last_day)
        last_day += relativedelta(weeks=+window_size_weeks)

    sliding_dates.append(window_dates)

    return sliding_dates


def get_all_ANM():
    df = pd.read_csv('fixedinputs/allANM.csv')
    return df


def get_predict_ANM(fileName):
    if fileName == None :
        return None
    else:
        df = pd.read_csv(fileName)
        next_anm_id = np.array(df['sub_center_id'])
        return next_anm_id


def get_campdate_level_df(df, online_day):
    columns = ['sub_center_id', 'camp_id', 'ANC_Mother Id', 'ANC_ANC1 Date', 'ANC_ANC2 Date', 'ANC_ANC3 Date',
               'ANC_ANC4 Date',
               'ANC_ANC1 BP Sys', 'ANC_ANC1 BP Dia', 'ANC_ANC2 BP Sys', 'ANC_ANC2 BP Dia', 'ANC_ANC3 BP Sys',
               'ANC_ANC3 BP Dia', 'ANC_ANC4 BP Sys', 'ANC_ANC4 BP Dia',
               'ANC_ANC1 Urine Albumin', 'ANC_ANC2 Urine Albumin', 'ANC_ANC3 Urine Albumin', 'ANC_ANC4 Urine Albumin',
               'ANC_ANC1 Urine Sugar', 'ANC_ANC2 Urine Sugar', 'ANC_ANC3 Urine Sugar', 'ANC_ANC4 Urine Sugar',
               'ANC_ANC1 HB', 'ANC_ANC2 HB', 'ANC_ANC3 HB', 'ANC_ANC4 HB',
               'ANC_ANC1 Weight', 'ANC_ANC2 Weight', 'ANC_ANC3 Weight', 'ANC_ANC4 Weight',
               'ANC_ANC1 Blood Sugar', 'ANC_ANC2 Blood Sugar', 'ANC_ANC3 Blood Sugar', 'ANC_ANC4 Blood Sugar',
               'ANC_ANC1 Fetal Heart Rate', 'ANC_ANC2 Fetal Heart Rate', 'ANC_ANC3 Fetal Heart Rate',
               'ANC_ANC4 Fetal Heart Rate']
    dfmod = df.loc[df['ANC_Mother Id'].notnull(), columns]

    newcolumns = ['date', 'BPsys', 'BPdia', 'urinesugar', 'albumin', 'hb', 'weight', 'blood_sugar', 'fetalhr']
    oldcolumns = [['ANC_ANC1 Date', 'ANC_ANC2 Date', 'ANC_ANC3 Date', 'ANC_ANC4 Date'],
                  ['ANC_ANC1 BP Sys', 'ANC_ANC2 BP Sys', 'ANC_ANC3 BP Sys', 'ANC_ANC4 BP Sys'],
                  ['ANC_ANC1 BP Dia', 'ANC_ANC2 BP Dia', 'ANC_ANC3 BP Dia', 'ANC_ANC4 BP Dia'],
                  ['ANC_ANC1 Urine Sugar', 'ANC_ANC2 Urine Sugar', 'ANC_ANC3 Urine Sugar', 'ANC_ANC4 Urine Sugar'],
                  ['ANC_ANC1 Urine Albumin', 'ANC_ANC2 Urine Albumin', 'ANC_ANC3 Urine Albumin',
                   'ANC_ANC4 Urine Albumin'],
                  ['ANC_ANC1 HB', 'ANC_ANC2 HB', 'ANC_ANC3 HB', 'ANC_ANC4 HB'],
                  ['ANC_ANC1 Weight', 'ANC_ANC2 Weight', 'ANC_ANC3 Weight', 'ANC_ANC4 Weight'],
                  ['ANC_ANC1 Blood Sugar', 'ANC_ANC2 Blood Sugar', 'ANC_ANC3 Blood Sugar', 'ANC_ANC4 Blood Sugar'],
                  ['ANC_ANC1 Fetal Heart Rate', 'ANC_ANC2 Fetal Heart Rate', 'ANC_ANC3 Fetal Heart Rate',
                   'ANC_ANC4 Fetal Heart Rate']]

    df2 = arrangeDate(dfmod, newcolumns, oldcolumns)
    df3 = afterSystemOnline(df2, online_day)

    mergeDates = MergeDates(online_day=online_day)
    df4 = mergeDates.addClusters(df3, df)

    return df4


def filter_new_data(df, online_day):
    df['ANC_ANC1 Date'] = pd.to_datetime(df['ANC_ANC1 Date'])
    df['ANC_ANC2 Date'] = pd.to_datetime(df['ANC_ANC2 Date'])
    df['ANC_ANC3 Date'] = pd.to_datetime(df['ANC_ANC3 Date'])
    df['ANC_ANC4 Date'] = pd.to_datetime(df['ANC_ANC4 Date'])
    df['Delivery_Delivery Date'] = pd.to_datetime(df['Delivery_Delivery Date'])
    filt = (df['ANC_ANC1 Date'] >= online_day) | (df['ANC_ANC2 Date'] >= online_day) | (
            df['ANC_ANC3 Date'] >= online_day) | (df['ANC_ANC4 Date'] >= online_day) | (
            df['Delivery_Delivery Date'] >= online_day)
    filt_df = df[filt]
    return filt_df


