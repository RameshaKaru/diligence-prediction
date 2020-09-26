import pandas as pd
import numpy as np
import yaml
import datetime
from dateutil.relativedelta import relativedelta
from tensorflow import keras as keras

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


def get_saved_model():
    """
    Obtains the saved neural network model

    Returns
    -------
    keras model

    """
    lstm_model_cm = keras.models.load_model('models/regression_model')
    return lstm_model_cm


def get_cluster_centers():
    """
    Obtains the saved cmeans cluster centers

    Returns
    -------
    numpy array with cmeans centers of 18 dimensions

    """
    df = pd.read_csv('fixedinputs/cmeans_centers.csv')
    center1 = np.array(df["cluster1"])
    center2 = np.array(df["cluster2"])
    cntr = np.append(center1.reshape(center1.shape[0], 1), center2.reshape(center2.shape[0], 1), axis=1)
    return cntr.T


def get_configs():
    """
    Obtains the configurations from the config file

    Returns
    -------
    dictionary with configurations

    """
    with open('config.yaml', 'r') as stream:
        try:
            configs = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    return configs


def get_all_ANM():
    """
    Gets a dataframe with the IDs of all sub centers from the saved file

    Returns
    -------
    dataframe with all sub center ids

    """
    df = pd.read_csv('fixedinputs/allANM.csv')
    return df


def get_predict_ANM(fileName):
    """
    Gets the sub center ids of the next prediction window from the configurations

    Parameters
    ----------
    fileName : string or None
        file name of the csv file with sub center ids of the next prediction window

    Returns
    -------
    numpy array of sub center ids

    """

    if fileName == None :
        return None
    else:
        df = pd.read_csv(fileName)
        next_anm_id = np.array(df['sub_center_id'])
        return next_anm_id


def get_campdate_level_df(df, online_day):
    """
    Processs the dataframes in a format suitable to input to the short range rules

    Parameters
    ----------
    df : dataframe
        dataframe to be processed
    online_day : date
        starting date

    Returns
    -------
    processed dataframe

    """

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
    """
    Filter the new data by selecting only the records after a given date

    Parameters
    ----------
    df : dataframe
        dataframe to be processed
    online_day : date
        starting date

    Returns
    -------
    filtered dataframe

    """
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

def get_ignore_dates(ignore_dates):
    """
    Obtains the ignore date ranges from the configurations and convert them to datetime objects

    Parameters
    ----------
    ignore_dates : list of dictionaries
        contains the date ranges which should be ignored

    Returns
    -------
    converted datetime objects

    """

    if ignore_dates is not None:
        for i in range(len(ignore_dates)):
            ignore_dates[i]['start_date'] = datetime.datetime.strptime(ignore_dates[i]['start_date'],
                                                                        '%Y, %m, %d').date()
            ignore_dates[i]['end_date'] = datetime.datetime.strptime(ignore_dates[i]['end_date'],
                                                                      '%Y, %m, %d').date()
    return ignore_dates

def get_overlap(start1, end1, start2, end2):
    """
    Obtains the overlap between 2 date ranges

    Parameters
    ----------
    start1 : date
        starting date of the 1st date range
    end1 : date
        ending date of the 1st date range
    start2 : date
        starting date of the 2nd date range
    end2 : date
        ending date of the 2nd date range

    Returns
    -------
    number of overlapped days

    """
    latest_start = max(start1, start2)
    earliest_end = min(end1, end2)
    delta = (earliest_end - latest_start).days
    overlap = max(0, delta)

    return overlap

def get_no_ignore_sliding_dates(end_day, start_end_day):
    """
    Obtains the sliding window dates when ignore date ranges are not given in configs

    Parameters
    ----------
    end_day : date
        ending date of whole data set
    start_end_day : date
        earliest possible end dte for a slide

    Returns
    -------
    list of sliding window dates

    """
    sliding_dates_list = []
    while len(sliding_dates_list) < 6:
        sliding_dates_list.append(end_day)
        end_day = end_day + relativedelta(weeks=-4)
        if end_day < start_end_day:
            print("Not enough data frames. Please add data of more than 1 year")
            break
    sliding_dates_list.reverse()
    return sliding_dates_list


