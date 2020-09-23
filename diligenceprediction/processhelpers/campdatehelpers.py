import pandas as pd
import numpy as np


# Rearranges the data: to concat 4 ANC dates and medical data relevant to 4 dates
def arrangeDate(df, newcolumns, oldcolumns):
    columns = ['sub_center_id', 'camp_id', 'ANC_Mother Id']
    columns = np.concatenate((columns, newcolumns))
    df2 = pd.DataFrame(columns=columns)
    df2['sub_center_id'] = pd.concat(
        [df['sub_center_id'], df['sub_center_id'], df['sub_center_id'], df['sub_center_id']], ignore_index=True)
    df2['camp_id'] = pd.concat([df['camp_id'], df['camp_id'], df['camp_id'], df['camp_id']], ignore_index=True)
    df2['ANC_Mother Id'] = pd.concat(
        [df['ANC_Mother Id'], df['ANC_Mother Id'], df['ANC_Mother Id'], df['ANC_Mother Id']], ignore_index=True)

    for i in range(len(newcolumns)):
        df2[newcolumns[i]] = pd.concat([df[oldcolumns[i][j]] for j in range(4)], ignore_index=True)

    return df2


def afterSystemOnline(df3, online_day, column='date'):
    df3[column] = pd.to_datetime(df3[column])
    df4 = df3[(df3[column] >= online_day)].reset_index()
    return df4


# get all camp ids and ANM ids in a given df
def getANMCampIds(df):
    df2 = df.groupby(['sub_center_id']).agg(
        camp_id=pd.NamedAgg(column='camp_id', aggfunc=lambda x: list(set(x)))
    ).reset_index()

    return list(df2['sub_center_id']), list(df2['camp_id'])


# get all camp ids
def getAllCampIds(df):
    return list(set(df['camp_id']))


# Rearranges the data: to concat 4 ANC dates and medical data relevant to 4 dates
def arrangeDate(df, newcolumns, oldcolumns):
    columns = ['sub_center_id', 'camp_id', 'ANC_Mother Id']
    columns = np.concatenate((columns, newcolumns))
    df2 = pd.DataFrame(columns=columns)
    df2['sub_center_id'] = pd.concat(
        [df['sub_center_id'], df['sub_center_id'], df['sub_center_id'], df['sub_center_id']], ignore_index=True)
    df2['camp_id'] = pd.concat([df['camp_id'], df['camp_id'], df['camp_id'], df['camp_id']], ignore_index=True)
    df2['ANC_Mother Id'] = pd.concat(
        [df['ANC_Mother Id'], df['ANC_Mother Id'], df['ANC_Mother Id'], df['ANC_Mother Id']], ignore_index=True)

    for i in range(len(newcolumns)):
        df2[newcolumns[i]] = pd.concat([df[oldcolumns[i][j]] for j in range(4)], ignore_index=True)

    return df2


