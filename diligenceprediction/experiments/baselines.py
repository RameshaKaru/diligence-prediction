# -*- coding: utf-8 -*-
"""baselines.ipynb

Original file is located at
    https://colab.research.google.com/drive/1xLLIAYgGU1v3mzdJYV7etZ9V0rgxiw2G

"""

import numpy as np
import datetime
from dateutil.relativedelta import relativedelta
from tensorflow import keras

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pandas as pd
import matplotlib as plt
from sklearn.model_selection import train_test_split
import time



import tensorflow as tf
from tensorflow.python import tf2

if not tf2.enabled():
    import tensorflow.compat.v2 as tf

    tf.enable_v2_behavior()
    assert tf2.enabled()

import tensorflow_probability as tfp



"""Rules to identify diligency (18)"""

def getBPsdRuleData(df,dummydf):
  """
    Rule: BP values: If majority (>70%) of measurements are 120/80 or 110/70
    
    Step 1. Processes the data in the given timeframe using the rule
    Step 2. Finds the percentages of non-diligence readings per ANM accoriding to the rule
    Step 3. Maps the percentages to non-diligence probabilities using KDE and outputs it

    Parameters
    ----------
    df : pandas DataFrame
        Dataframe to be processed
    dummydf : pandas DataFrame
        Dataframe with all the ANMs

    Returns
    -------
    list
        a list of non-diligence probabilities of ANMs
    """
  # Step 1
  grouped = df.groupby(['sub_center_id', 'camp_id','cluster_date'])
  bp = grouped.agg({'ANC_Mother Id': lambda x: len(list(x)),
                    'BPsys': lambda x: list(x),
                    'BPdia': lambda x: list(x)}).reset_index()

  # Step 2
  badcount = []
  totalcount = []
  for k in range(len(bp)):
    c = 0
    tot = 0
    for i in range(len(bp.loc[k,'BPsys'])):
      if (bp.loc[k,'BPsys'][i]==120) and (bp.loc[k,'BPdia'][i]==80) :
        c += 1
        tot +=1
      elif (bp.loc[k,'BPsys'][i]==110) and (bp.loc[k,'BPdia'][i]==70) :
        c += 1
        tot +=1
      elif (bp.loc[k,'BPsys'][i]>7) or (bp.loc[k,'BPdia'][i]>7):
        tot +=1
    badcount.append(c)
    totalcount.append(tot)

  badbpcount = pd.Series(badcount)
  bp = bp.assign(badbpcount=badbpcount.values)
  totalbpcount = pd.Series(totalcount)
  bp = bp.assign(totalbpcount=totalbpcount.values)
  bp['totalbpcount'].fillna(0, inplace=True)
  bp['suspicious_percentage'] = bp.apply(lambda row: row['badbpcount']/row['totalbpcount'] if row['totalbpcount'] != 0 else 0, axis=1)
  
  anmgrouped = bp.groupby(['sub_center_id'])
  bpanm = anmgrouped.agg(
    Num_camps = pd.NamedAgg(column = 'suspicious_percentage', aggfunc= lambda x: len(list(x))),
    Tot_num_patients = pd.NamedAgg(column = 'ANC_Mother Id', aggfunc= sum),
    suspicious_percentage = pd.NamedAgg(column = 'suspicious_percentage', aggfunc= lambda x: list(x)),
    cluster_dates = pd.NamedAgg(column = 'cluster_date', aggfunc= lambda x: list(x))
  ).reset_index()

  # Step 3
  fraud_probabilities = []
  for i in range(len(bpanm)):
    fraud_prob = []
    for j in range(len(bpanm.loc[i,'suspicious_percentage'])):
      x = bpanm.loc[i,'suspicious_percentage'][j]
      #prob = func_get_prob_mass_trans(bpsd_fit,0,x)
      fraud_prob.append(x)
    fraud_probabilities.append(np.mean(fraud_prob))

  fraud_probabilities = pd.Series(fraud_probabilities)
  bpanm = bpanm.assign(fraud_probabilities=fraud_probabilities.values)

  # merge with dummy dataframe with all ANM
  df_merge = pd.merge(dummydf, bpanm, on='sub_center_id', how='outer')
  df_merge['fraud_probabilities'].fillna(df_merge['fraud_probabilities'].mean(), inplace=True)

  return list(df_merge['fraud_probabilities'])

def getModfUrineRule(df,dummydf):
  """
    RULE: Urine test: greater than 90% of values marked as “Absent”; more than 10% should be Present or Test Not Done
    
    Step 1. Processes the data in the given timeframe using the rule
    Step 2. Finds the percentages of non-diligence readings per ANM accoriding to the rule
    Step 3. Maps the percentages to non-diligence probabilities using KDE and outputs it

    Parameters
    ----------
    df : pandas DataFrame
        Dataframe to be processed
    dummydf : pandas DataFrame
        Dataframe with all the ANMs

    Returns
    -------
    list
        a list of non-diligence probabilities of ANMs
    """
  # Step 1
  grouped = df.groupby(['sub_center_id', 'camp_id', 'cluster_date'])
  ur = grouped.agg({'ANC_Mother Id': lambda x: len(list(x)),
                    'urinesugar': lambda x: list(x),
                    'albumin':lambda x: list(x)}).reset_index()

  # Step 2
  absentcount = []
  totcount = []
  for k in range(len(ur)):
    abcountanm = 0
    totcountanm = 0
    for i in range(len(ur.loc[k,'albumin'])):
      if (ur.loc[k,'albumin'][i]=='False') and (ur.loc[k,'urinesugar'][i]=='False') :
        abcountanm += 1
        totcountanm += 1
      elif (ur.loc[k,'albumin'][i]=='False') and (ur.loc[k,'urinesugar'][i]=='True') :
        totcountanm += 1
      elif (ur.loc[k,'albumin'][i]=='True') and (ur.loc[k,'urinesugar'][i]=='False') :
        totcountanm += 1
      elif (ur.loc[k,'albumin'][i]=='True') and (ur.loc[k,'urinesugar'][i]=='True') :
        totcountanm += 1
    absentcount.append(abcountanm)
    totcount.append(totcountanm)

  absenturcount = pd.Series(absentcount)
  ur = ur.assign(absenturcount=absenturcount.values)
  toturcount = pd.Series(totcount)
  ur = ur.assign(toturcount=toturcount.values)
  ur['falsepercentage'] = ur.apply(lambda row: row['absenturcount']/row['toturcount'] if row['toturcount'] != 0 else 0, axis=1)

  anmgrouped = ur.groupby(['sub_center_id'])
  uranm = anmgrouped.agg(
    Num_camps = pd.NamedAgg(column = 'falsepercentage', aggfunc= lambda x: len(list(x))),
    Tot_num_patients = pd.NamedAgg(column = 'ANC_Mother Id', aggfunc= sum),
    false_percentage = pd.NamedAgg(column = 'falsepercentage', aggfunc= lambda x: list(x)),
    cluster_dates = pd.NamedAgg(column = 'cluster_date', aggfunc= lambda x: list(x))
  ).reset_index()
  uranm.to_csv("urine.csv")
  # Step 3
  fraud_probabilities = []
  for i in range(len(uranm)):
    fraud_prob = []
    for j in range(len(uranm.loc[i,'false_percentage'])):
      x = uranm.loc[i,'false_percentage'][j]
      #prob = func_get_prob_mass_trans(urine_fit,0,x)
      fraud_prob.append(x)
    fraud_probabilities.append(np.mean(fraud_prob))

  fraud_probabilities = pd.Series(fraud_probabilities)
  uranm = uranm.assign(fraud_probabilities=fraud_probabilities.values)

  # merge with dummy dataframe
  df_merge = pd.merge(dummydf, uranm, on='sub_center_id', how='outer')
  df_merge['fraud_probabilities'].fillna(df_merge['fraud_probabilities'].mean(), inplace=True)

  return list(df_merge['fraud_probabilities'])

def getBPhyperRule(df,dummydf):
  """
    "RULE: Proportion of patients with Hypertension (either BP Sys > 140 OR BP Dia > 90) < 5%"
    
    Step 1. Processes the data in the given timeframe using the rule
    Step 2. Finds the percentages of non-diligence readings per ANM accoriding to the rule
    Step 3. Maps the percentages to non-diligence probabilities using KDE and outputs it

    Parameters
    ----------
    df : pandas DataFrame
        Dataframe to be processed
    dummydf : pandas DataFrame
        Dataframe with all the ANMs

    Returns
    -------
    list
        a list of non-diligence probabilities of ANMs
    """
  # Step 1
  grouped = df.groupby(['sub_center_id', 'camp_id','cluster_date'])
  bp = grouped.agg({'ANC_Mother Id': lambda x: len(list(x)),
                    'BPsys': lambda x: list(x),
                    'BPdia': lambda x: list(x)}).reset_index()

  # Step 2
  badcount = []
  totalcount = []
  for k in range(len(bp)):
    c = 0
    tot = 0
    for i in range(len(bp.loc[k,'BPsys'])):
      if (bp.loc[k,'BPsys'][i] > 140) or (bp.loc[k,'BPdia'][i] > 90) :
        c += 1
        tot +=1
      elif (bp.loc[k,'BPsys'][i]>7) or (bp.loc[k,'BPdia'][i]>7):
        tot +=1
    badcount.append(c)
    totalcount.append(tot)

  badbpcount = pd.Series(badcount)
  bp = bp.assign(badbpcount=badbpcount.values)
  totalbpcount = pd.Series(totalcount)
  bp = bp.assign(totalbpcount=totalbpcount.values)
  bp['totalbpcount'].fillna(0, inplace=True)
  bp['hyper_percentage'] = bp.apply(lambda row: row['badbpcount']/row['totalbpcount']  if row['totalbpcount'] != 0 else 0, axis=1)

  anmgrouped = bp.groupby(['sub_center_id'])
  bpanm = anmgrouped.agg(
    Num_camps = pd.NamedAgg(column = 'hyper_percentage', aggfunc= lambda x: len(list(x))),
    Tot_num_patients = pd.NamedAgg(column = 'ANC_Mother Id', aggfunc= sum),
    hyper_percentage = pd.NamedAgg(column = 'hyper_percentage', aggfunc= lambda x: list(x)),
    dates = pd.NamedAgg(column = 'cluster_date', aggfunc= lambda x: list(x))
  ).reset_index()
  bpanm.to_csv("hyperbp.csv")
  # Step 3
  fraud_probabilities = []
  for i in range(len(bpanm)):
    fraud_prob = []
    for j in range(len(bpanm.loc[i,'hyper_percentage'])):
      x = bpanm.loc[i,'hyper_percentage'][j]
      #prob = func_get_prob_mass_trans(bphyper_fit,x,100)
      fraud_prob.append(x)
    fraud_probabilities.append(np.mean(fraud_prob))

  fraud_probabilities = pd.Series(fraud_probabilities)
  bpanm = bpanm.assign(fraud_probabilities=fraud_probabilities.values)

  # merge with dummy dataframe
  df_merge = pd.merge(dummydf, bpanm, on='sub_center_id', how='outer')
  df_merge['fraud_probabilities'].fillna(df_merge['fraud_probabilities'].mean(), inplace=True)

  return list(df_merge['fraud_probabilities'])

def getHBrule(df,dummydf):  
  """
    "RULE: Proportion of Patients with Anemia (defined as HB < 11) < 48% or >68%"
    
    Step 1. Processes the data in the given timeframe using the rule
    Step 2. Finds the percentages of non-diligence readings per ANM accoriding to the rule
    Step 3. Maps the percentages to non-diligence probabilities using KDE and outputs it

    Parameters
    ----------
    df : pandas DataFrame
        Dataframe to be processed
    dummydf : pandas DataFrame
        Dataframe with all the ANMs

    Returns
    -------
    list
        a list of non-diligence probabilities of ANMs
    """
  # Step 1
  grouped = df.groupby(['sub_center_id', 'camp_id','cluster_date'])
  bp = grouped.agg({'ANC_Mother Id': lambda x: len(list(x)),
                    'hb': lambda x: list(x)}).reset_index()

  # Step 2
  anaemiaCount = []
  totalRealCount = []
  for k in range(len(bp)):
    anaemia = 0
    totReal = 0
    for i in range(len(bp.loc[k,'hb'])):
      if (bp.loc[k,'hb'][i]< 11) and (bp.loc[k,'hb'][i]>5):
        anaemia += 1
        totReal += 1
      elif (bp.loc[k,'hb'][i]>5):
        totReal += 1
    
    anaemiaCount.append(anaemia)
    totalRealCount.append(totReal)

  bp = bp.assign(anaemiaCount=anaemiaCount)
  bp = bp.assign(totalRealCount=totalRealCount)
  bp['totalRealCount'].fillna(0, inplace=True)
  bp['anaemia_percentage'] = bp.apply(lambda row: row['anaemiaCount']/row['totalRealCount'] if row['totalRealCount'] != 0 else 0, axis=1)

  anmgrouped = bp.groupby(['sub_center_id'])
  bpanm = anmgrouped.agg(
    Num_camps = pd.NamedAgg(column = 'anaemia_percentage', aggfunc= lambda x: len(list(x))),
    Tot_num_patients = pd.NamedAgg(column = 'ANC_Mother Id', aggfunc= sum),
    anaemia_percentage = pd.NamedAgg(column = 'anaemia_percentage', aggfunc= lambda x: list(x)),
    dates = pd.NamedAgg(column = 'cluster_date', aggfunc= lambda x: list(x))
  ).reset_index()

  # Step 3
  fraud_probabilities = []
  for i in range(len(bpanm)):
    fraud_prob = []
    for j in range(len(bpanm.loc[i,'anaemia_percentage'])):
      x = bpanm.loc[i,'anaemia_percentage'][j]
      # if (x<50):
      #   probR = func_get_prob_mass_trans(hb50_fit,x,50)
      #   prob = probR[0]
      # elif (x>70):
      #   probR = func_get_prob_mass_trans(hb70_fit,70,x)
      #   prob = probR[0]
      # else:
      #   prob = 0.0

      fraud_prob.append(x)
    fraud_probabilities.append(np.mean(fraud_prob))

  fraud_probabilities = pd.Series(fraud_probabilities)
  bpanm = bpanm.assign(fraud_probabilities=fraud_probabilities.values)

  # merge with dummy dataframe
  df_merge = pd.merge(dummydf, bpanm, on='sub_center_id', how='outer')
  df_merge['fraud_probabilities'].fillna(df_merge['fraud_probabilities'].mean(), inplace=True)

  return list(df_merge['fraud_probabilities'])

def getHB7rule(df,dummydf):  
  """
    "RULE: Proportion of Patients with HB < 7 < 5%"
    
    Step 1. Processes the data in the given timeframe using the rule
    Step 2. Finds the percentages of non-diligence readings per ANM accoriding to the rule
    Step 3. Maps the percentages to non-diligence probabilities using KDE and outputs it

    Parameters
    ----------
    df : pandas DataFrame
        Dataframe to be processed
    dummydf : pandas DataFrame
        Dataframe with all the ANMs

    Returns
    -------
    list
        a list of non-diligence probabilities of ANMs
    """
  # Step 1
  grouped = df.groupby(['sub_center_id', 'camp_id','cluster_date'])
  bp = grouped.agg({'ANC_Mother Id': lambda x: len(list(x)),
                    'hb': lambda x: list(x)}).reset_index()

  # Step 2
  anaemiaCount = []
  totalRealCount = []
  for k in range(len(bp)):
    anaemia = 0
    totReal = 0
    for i in range(len(bp.loc[k,'hb'])):
      if (bp.loc[k,'hb'][i]< 7) and (bp.loc[k,'hb'][i]>5):
        anaemia += 1
        totReal += 1
      elif (bp.loc[k,'hb'][i]>5):
        totReal += 1
    
    anaemiaCount.append(anaemia)
    totalRealCount.append(totReal)

  bp = bp.assign(anaemiaCount=anaemiaCount)
  bp = bp.assign(totalRealCount=totalRealCount)
  bp['totalRealCount'].fillna(0, inplace=True)
  bp['anaemia_percentage'] = bp.apply(lambda row: row['anaemiaCount']/row['totalRealCount']  if row['totalRealCount'] != 0 else 0, axis=1)

  anmgrouped = bp.groupby(['sub_center_id'])
  bpanm = anmgrouped.agg(
    Num_camps = pd.NamedAgg(column = 'anaemia_percentage', aggfunc= lambda x: len(list(x))),
    Tot_num_patients = pd.NamedAgg(column = 'ANC_Mother Id', aggfunc= sum),
    anaemia_percentage = pd.NamedAgg(column = 'anaemia_percentage', aggfunc= lambda x: list(x)),
    dates = pd.NamedAgg(column = 'cluster_date', aggfunc= lambda x: list(x))
  ).reset_index()

  # Step 3
  fraud_probabilities = []
  for i in range(len(bpanm)):
    fraud_prob = []
    for j in range(len(bpanm.loc[i,'anaemia_percentage'])):
      x = bpanm.loc[i,'anaemia_percentage'][j]
      #prob = func_get_prob_mass_trans(hb7_fit,x,100)
      fraud_prob.append(x)
    fraud_probabilities.append(np.mean(fraud_prob))

  fraud_probabilities = pd.Series(fraud_probabilities)
  bpanm = bpanm.assign(fraud_probabilities=fraud_probabilities.values)

  # merge with dummy dataframe
  df_merge = pd.merge(dummydf, bpanm, on='sub_center_id', how='outer')
  df_merge['fraud_probabilities'].fillna(df_merge['fraud_probabilities'].mean(), inplace=True)

  return list(df_merge['fraud_probabilities'])

def bpcontradictionRule(df,dummydf):
  """
    "RULE: “No-Equipment/Entered-Data Contradiction” for blood pressure (No Equipment and then “Present/Absent for another patient)"
    
    Step 1. Processes the data in the given timeframe using the rule
    Step 2. Finds the percentages of non-diligence readings per ANM accoriding to the rule
    Step 3. Maps the percentages to non-diligence probabilities using KDE and outputs it

    Parameters
    ----------
    df : pandas DataFrame
        Dataframe to be processed
    dummydf : pandas DataFrame
        Dataframe with all the ANMs

    Returns
    -------
    list
        a list of non-diligence probabilities of ANMs
    """
  # Step 1
  grouped = df.groupby(['sub_center_id', 'camp_id','cluster_date'])
  bp = grouped.agg(
    Num_patients = pd.NamedAgg(column = 'ANC_Mother Id', aggfunc= lambda x: len(list(x))),
    Num_checked_patients = pd.NamedAgg(column = 'BPsys', aggfunc= lambda x: len(list(x.dropna()))),
    BPsys = pd.NamedAgg(column = 'BPsys', aggfunc= lambda x: list(x.dropna())),
    BPdia = pd.NamedAgg(column = 'BPdia', aggfunc= lambda x: list(x.dropna()))   
  ).reset_index()

  # Step 2
  noEquipCount = []
  totalRealCount = []
  for k in range(len(bp)):
    noEquip = 0
    totReal = 0
    for i in range(len(bp.loc[k,'BPsys'])):
      if (bp.loc[k,'BPsys'][i]==4):
        noEquip += 1
      elif (bp.loc[k,'BPsys'][i]>7):
        totReal += 1
    
    noEquipCount.append(noEquip)
    totalRealCount.append(totReal)

  bp = bp.assign(noEquipCount=noEquipCount)
  bp = bp.assign(totalRealCount=totalRealCount)
  bp['totalRealCount'].fillna(0, inplace=True)
  bp['suspicious_percentage'] = bp.apply(lambda row: row['noEquipCount']/row['Num_checked_patients']  if row['totalRealCount'] != 0 else 0, axis=1)

  anmgrouped = bp.groupby(['sub_center_id'])
  bpanm = anmgrouped.agg(
    Num_camps = pd.NamedAgg(column = 'suspicious_percentage', aggfunc= lambda x: len(list(x))),
    Tot_num_patients = pd.NamedAgg(column = 'Num_patients', aggfunc= sum),
    suspicious_percentage = pd.NamedAgg(column = 'suspicious_percentage', aggfunc= lambda x: list(x)),
    dates = pd.NamedAgg(column = 'cluster_date', aggfunc= lambda x: list(x))
  ).reset_index()

  # Step 3
  fraud_probabilities = []
  for i in range(len(bpanm)):
    fraud_prob = []
    for j in range(len(bpanm.loc[i,'suspicious_percentage'])):
      x = bpanm.loc[i,'suspicious_percentage'][j]
      #prob = func_get_prob_mass_trans(bp_contra_fit,0,x)
      fraud_prob.append(x)
    fraud_probabilities.append(np.mean(fraud_prob))

  fraud_probabilities = pd.Series(fraud_probabilities)
  bpanm = bpanm.assign(fraud_probabilities=fraud_probabilities.values)

  # merge with dummy dataframe
  df_merge = pd.merge(dummydf, bpanm, on='sub_center_id', how='outer')
  df_merge['fraud_probabilities'].fillna(df_merge['fraud_probabilities'].mean(), inplace=True)

  return list(df_merge['fraud_probabilities'])

def weightcontradictionRule(df,dummydf):
  """
    "RULE: “No-Equipment/Entered-Data Contradiction” for weight (No Equipment and then “Present/Absent for another patient)"

    Step 1. Processes the data in the given timeframe using the rule
    Step 2. Finds the percentages of non-diligence readings per ANM accoriding to the rule
    Step 3. Maps the percentages to non-diligence probabilities using KDE and outputs it

    Parameters
    ----------
    df : pandas DataFrame
        Dataframe to be processed
    dummydf : pandas DataFrame
        Dataframe with all the ANMs

    Returns
    -------
    list
        a list of non-diligence probabilities of ANMs
    """
  # Step 1
  grouped = df.groupby(['sub_center_id', 'camp_id','cluster_date'])
  bp = grouped.agg(
    Num_patients = pd.NamedAgg(column = 'ANC_Mother Id', aggfunc= lambda x: len(list(x))),
    Num_checked_patients = pd.NamedAgg(column = 'weight', aggfunc= lambda x: len(list(x.dropna()))),
    weight = pd.NamedAgg(column = 'weight', aggfunc= lambda x: list(x.dropna()))   
  ).reset_index()

  #Step 2
  noEquipCount = []
  totalRealCount = []
  for k in range(len(bp)):
    noEquip = 0
    totReal = 0
    for i in range(len(bp.loc[k,'weight'])):
      if (bp.loc[k,'weight'][i]==4):
        noEquip += 1
      elif (bp.loc[k,'weight'][i]>7):
        totReal += 1
    
    noEquipCount.append(noEquip)
    totalRealCount.append(totReal)

  bp = bp.assign(noEquipCount=noEquipCount)
  bp = bp.assign(totalRealCount=totalRealCount)
  bp['totalRealCount'].fillna(0, inplace=True)
  bp['suspicious_percentage'] = bp.apply(lambda row: row['noEquipCount']/row['Num_checked_patients']  if row['totalRealCount'] != 0 else 0, axis=1)

  anmgrouped = bp.groupby(['sub_center_id'])
  bpanm = anmgrouped.agg(
    Num_camps = pd.NamedAgg(column = 'suspicious_percentage', aggfunc= lambda x: len(list(x))),
    Tot_num_patients = pd.NamedAgg(column = 'Num_patients', aggfunc= sum),
    suspicious_percentage = pd.NamedAgg(column = 'suspicious_percentage', aggfunc= lambda x: list(x)),
    dates = pd.NamedAgg(column = 'cluster_date', aggfunc= lambda x: list(x))
  ).reset_index()

  # Step 3
  fraud_probabilities = []
  for i in range(len(bpanm)):
    fraud_prob = []
    for j in range(len(bpanm.loc[i,'suspicious_percentage'])):
      x = bpanm.loc[i,'suspicious_percentage'][j]
      #prob = func_get_prob_mass_trans(weight_contra_fit,0,x)
      fraud_prob.append(x)
    fraud_probabilities.append(np.mean(fraud_prob))

  fraud_probabilities = pd.Series(fraud_probabilities)
  bpanm = bpanm.assign(fraud_probabilities=fraud_probabilities.values)

  # merge with dummy dataframe
  df_merge = pd.merge(dummydf, bpanm, on='sub_center_id', how='outer')
  df_merge['fraud_probabilities'].fillna(df_merge['fraud_probabilities'].mean(), inplace=True)

  return list(df_merge['fraud_probabilities'])

def hbcontradictionRule(df,dummydf):
  """
    "RULE: “No-Equipment/Entered-Data Contradiction” for haemoglobin (No Equipment and then “Present/Absent for another patient)"
    
    Step 1. Processes the data in the given timeframe using the rule
    Step 2. Finds the percentages of non-diligence readings per ANM accoriding to the rule
    Step 3. Maps the percentages to non-diligence probabilities using KDE and outputs it

    Parameters
    ----------
    df : pandas DataFrame
        Dataframe to be processed
    dummydf : pandas DataFrame
        Dataframe with all the ANMs

    Returns
    -------
    list
        a list of non-diligence probabilities of ANMs
    """

  # Step 1
  grouped = df.groupby(['sub_center_id', 'camp_id','cluster_date'])
  bp = grouped.agg(
    Num_patients = pd.NamedAgg(column = 'ANC_Mother Id', aggfunc= lambda x: len(list(x))),
    Num_checked_patients = pd.NamedAgg(column = 'hb', aggfunc= lambda x: len(list(x.dropna()))),
    hb = pd.NamedAgg(column = 'hb', aggfunc= lambda x: list(x.dropna()))   
  ).reset_index()

  # Step 2
  noEquipCount = []
  totalRealCount = []
  for k in range(len(bp)):
    noEquip = 0
    totReal = 0
    for i in range(len(bp.loc[k,'hb'])):
      if (bp.loc[k,'hb'][i]==4):
        noEquip += 1
      elif (bp.loc[k,'hb'][i]>5):
        totReal += 1
    
    noEquipCount.append(noEquip)
    totalRealCount.append(totReal)

  bp = bp.assign(noEquipCount=noEquipCount)
  bp = bp.assign(totalRealCount=totalRealCount)
  bp['totalRealCount'].fillna(0, inplace=True)
  bp['suspicious_percentage'] = bp.apply(lambda row: row['noEquipCount']/row['Num_checked_patients']  if row['totalRealCount'] != 0 else 0, axis=1)

  anmgrouped = bp.groupby(['sub_center_id'])
  bpanm = anmgrouped.agg(
    Num_camps = pd.NamedAgg(column = 'suspicious_percentage', aggfunc= lambda x: len(list(x))),
    Tot_num_patients = pd.NamedAgg(column = 'Num_patients', aggfunc= sum),
    suspicious_percentage = pd.NamedAgg(column = 'suspicious_percentage', aggfunc= lambda x: list(x)),
    dates = pd.NamedAgg(column = 'cluster_date', aggfunc= lambda x: list(x))
  ).reset_index()

  # Step 3
  fraud_probabilities = []
  for i in range(len(bpanm)):
    fraud_prob = []
    for j in range(len(bpanm.loc[i,'suspicious_percentage'])):
      x = bpanm.loc[i,'suspicious_percentage'][j]
      #prob = func_get_prob_mass_trans(hb_contra_fit,0,x)
      fraud_prob.append(x)
    fraud_probabilities.append(np.mean(fraud_prob))

  fraud_probabilities = pd.Series(fraud_probabilities)
  bpanm = bpanm.assign(fraud_probabilities=fraud_probabilities.values)

  # merge with dummy dataframe
  df_merge = pd.merge(dummydf, bpanm, on='sub_center_id', how='outer')
  df_merge['fraud_probabilities'].fillna(df_merge['fraud_probabilities'].mean(), inplace=True)

  return list(df_merge['fraud_probabilities'])

def bloodsugarcontradictionRule(df,dummydf):
  """
    "RULE: “No-Equipment/Entered-Data Contradiction” for blood sugar (No Equipment and then “Present/Absent for another patient)"
    
    Step 1. Processes the data in the given timeframe using the rule
    Step 2. Finds the percentages of non-diligence readings per ANM accoriding to the rule
    Step 3. Maps the percentages to non-diligence probabilities using KDE and outputs it

    Parameters
    ----------
    df : pandas DataFrame
        Dataframe to be processed
    dummydf : pandas DataFrame
        Dataframe with all the ANMs

    Returns
    -------
    list
        a list of non-diligence probabilities of ANMs
    """
  # Step 1
  grouped = df.groupby(['sub_center_id', 'camp_id','cluster_date'])
  bp = grouped.agg(
    Num_patients = pd.NamedAgg(column = 'ANC_Mother Id', aggfunc= lambda x: len(list(x))),
    Num_checked_patients = pd.NamedAgg(column = 'blood_sugar', aggfunc= lambda x: len(list(x.dropna()))),
    blood_sugar = pd.NamedAgg(column = 'blood_sugar', aggfunc= lambda x: list(x.dropna()))   
  ).reset_index()

  # Step 2
  noEquipCount = []
  totalRealCount = []
  for k in range(len(bp)):
    noEquip = 0
    totReal = 0
    for i in range(len(bp.loc[k,'blood_sugar'])):
      if (bp.loc[k,'blood_sugar'][i]==4):
        noEquip += 1
      elif (bp.loc[k,'blood_sugar'][i]>7):
        totReal += 1
    
    noEquipCount.append(noEquip)
    totalRealCount.append(totReal)

  bp = bp.assign(noEquipCount=noEquipCount)
  bp = bp.assign(totalRealCount=totalRealCount)
  bp['totalRealCount'].fillna(0, inplace=True)
  bp['suspicious_percentage'] = bp.apply(lambda row: row['noEquipCount']/row['Num_checked_patients'] if row['totalRealCount'] != 0 else 0, axis=1)

  anmgrouped = bp.groupby(['sub_center_id'])
  bpanm = anmgrouped.agg(
    Num_camps = pd.NamedAgg(column = 'suspicious_percentage', aggfunc= lambda x: len(list(x))),
    Tot_num_patients = pd.NamedAgg(column = 'Num_patients', aggfunc= sum),
    suspicious_percentage = pd.NamedAgg(column = 'suspicious_percentage', aggfunc= lambda x: list(x)),
    dates = pd.NamedAgg(column = 'cluster_date', aggfunc= lambda x: list(x))
  ).reset_index()

  # Step 3
  fraud_probabilities = []
  for i in range(len(bpanm)):
    fraud_prob = []
    for j in range(len(bpanm.loc[i,'suspicious_percentage'])):
      x = bpanm.loc[i,'suspicious_percentage'][j]
      #prob = func_get_prob_mass_trans(bsugar_contra_fit,0,x)
      fraud_prob.append(x)
    fraud_probabilities.append(np.mean(fraud_prob))

  fraud_probabilities = pd.Series(fraud_probabilities)
  bpanm = bpanm.assign(fraud_probabilities=fraud_probabilities.values)

  # merge with dummy dataframe
  df_merge = pd.merge(dummydf, bpanm, on='sub_center_id', how='outer')
  df_merge['fraud_probabilities'].fillna(df_merge['fraud_probabilities'].mean(), inplace=True)

  return list(df_merge['fraud_probabilities'])

def fetalhrcontradictionRule(df,dummydf):
  """
    "RULE: “No-Equipment/Entered-Data Contradiction” for fetal heart rate (No Equipment and then “Present/Absent for another patient)"
    
    Step 1. Processes the data in the given timeframe using the rule
    Step 2. Finds the percentages of non-diligence readings per ANM accoriding to the rule
    Step 3. Maps the percentages to non-diligence probabilities using KDE and outputs it

    Parameters
    ----------
    df : pandas DataFrame
        Dataframe to be processed
    dummydf : pandas DataFrame
        Dataframe with all the ANMs

    Returns
    -------
    list
        a list of non-diligence probabilities of ANMs
    """
  # Step 1
  grouped = df.groupby(['sub_center_id', 'camp_id','cluster_date'])
  bp = grouped.agg(
    Num_patients = pd.NamedAgg(column = 'ANC_Mother Id', aggfunc= lambda x: len(list(x))),
    Num_checked_patients = pd.NamedAgg(column = 'fetalhr', aggfunc= lambda x: len(list(x.dropna()))),
    fetalhr = pd.NamedAgg(column = 'fetalhr', aggfunc= lambda x: list(x.dropna()))   
  ).reset_index()

  # Step 2
  noEquipCount = []
  totalRealCount = []
  for k in range(len(bp)):
    noEquip = 0
    totReal = 0
    for i in range(len(bp.loc[k,'fetalhr'])):
      if (bp.loc[k,'fetalhr'][i]==4):
        noEquip += 1
      elif (bp.loc[k,'fetalhr'][i]>7):
        totReal += 1
    
    noEquipCount.append(noEquip)
    totalRealCount.append(totReal)

  bp = bp.assign(noEquipCount=noEquipCount)
  bp = bp.assign(totalRealCount=totalRealCount)
  bp['totalRealCount'].fillna(0, inplace=True)
  bp['suspicious_percentage'] = bp.apply(lambda row: row['noEquipCount']/row['Num_checked_patients'] if row['totalRealCount'] != 0 else 0, axis=1)

  anmgrouped = bp.groupby(['sub_center_id'])
  bpanm = anmgrouped.agg(
    Num_camps = pd.NamedAgg(column = 'suspicious_percentage', aggfunc= lambda x: len(list(x))),
    Tot_num_patients = pd.NamedAgg(column = 'Num_patients', aggfunc= sum),
    suspicious_percentage = pd.NamedAgg(column = 'suspicious_percentage', aggfunc= lambda x: list(x)),
    dates = pd.NamedAgg(column = 'cluster_date', aggfunc= lambda x: list(x))
  ).reset_index()

  # Step 3
  fraud_probabilities = []
  for i in range(len(bpanm)):
    fraud_prob = []
    for j in range(len(bpanm.loc[i,'suspicious_percentage'])):
      x = bpanm.loc[i,'suspicious_percentage'][j]
      #prob = func_get_prob_mass_trans(fetalhr_contra_fit,0,x)
      fraud_prob.append(x)
    fraud_probabilities.append(np.mean(fraud_prob))

  fraud_probabilities = pd.Series(fraud_probabilities)
  bpanm = bpanm.assign(fraud_probabilities=fraud_probabilities.values)

  # merge with dummy dataframe
  df_merge = pd.merge(dummydf, bpanm, on='sub_center_id', how='outer')
  df_merge['fraud_probabilities'].fillna(df_merge['fraud_probabilities'].mean(), inplace=True)

  return list(df_merge['fraud_probabilities'])

def urinesugarcontradictionRule(df,dummydf):
  """
    "RULE: “No-Equipment/Entered-Data Contradiction” for Urine Test Sugar (No Equipment and then “Present/Absent for another patient)"
    
    Step 1. Processes the data in the given timeframe using the rule
    Step 2. Finds the percentages of non-diligence readings per ANM accoriding to the rule
    Step 3. Maps the percentages to non-diligence probabilities using KDE and outputs it

    Parameters
    ----------
    df : pandas DataFrame
        Dataframe to be processed
    dummydf : pandas DataFrame
        Dataframe with all the ANMs

    Returns
    -------
    list
        a list of non-diligence probabilities of ANMs
    """
  # Step 1
  grouped = df.groupby(['sub_center_id', 'camp_id','cluster_date'])
  bp = grouped.agg(
    Num_patients = pd.NamedAgg(column = 'ANC_Mother Id', aggfunc= lambda x: len(list(x))),
    Num_checked_patients = pd.NamedAgg(column = 'urinesugar', aggfunc= lambda x: len(list(x.dropna()))),
    urinesugar = pd.NamedAgg(column = 'urinesugar', aggfunc= lambda x: list(x.dropna()))   
  ).reset_index()

  # Step 2
  noEquipCount = []
  totalRealCount = []
  for k in range(len(bp)):
    noEquip = 0
    totReal = 0
    for i in range(len(bp.loc[k,'urinesugar'])):
      if (bp.loc[k,'urinesugar'][i]=='4'):
        noEquip += 1
      elif (bp.loc[k,'urinesugar'][i]=='True') or (bp.loc[k,'urinesugar'][i]=='False'):
        totReal += 1
    
    noEquipCount.append(noEquip)
    totalRealCount.append(totReal)

  bp = bp.assign(noEquipCount=noEquipCount)
  bp = bp.assign(totalRealCount=totalRealCount)
  bp['totalRealCount'].fillna(0, inplace=True)
  bp['suspicious_percentage'] = bp.apply(lambda row: row['noEquipCount']/row['Num_checked_patients']  if row['totalRealCount'] != 0 else 0, axis=1)

  anmgrouped = bp.groupby(['sub_center_id'])
  bpanm = anmgrouped.agg(
    Num_camps = pd.NamedAgg(column = 'suspicious_percentage', aggfunc= lambda x: len(list(x))),
    Tot_num_patients = pd.NamedAgg(column = 'Num_patients', aggfunc= sum),
    suspicious_percentage = pd.NamedAgg(column = 'suspicious_percentage', aggfunc= lambda x: list(x)),
    dates = pd.NamedAgg(column = 'cluster_date', aggfunc= lambda x: list(x))
  ).reset_index()

  # Step 3
  fraud_probabilities = []
  for i in range(len(bpanm)):
    fraud_prob = []
    for j in range(len(bpanm.loc[i,'suspicious_percentage'])):
      x = bpanm.loc[i,'suspicious_percentage'][j]
      #prob = func_get_prob_mass_trans(usugar_contra_fit,0,x)
      fraud_prob.append(x)
    fraud_probabilities.append(np.mean(fraud_prob))

  fraud_probabilities = pd.Series(fraud_probabilities)
  bpanm = bpanm.assign(fraud_probabilities=fraud_probabilities.values)

  # merge with dummy dataframe
  df_merge = pd.merge(dummydf, bpanm, on='sub_center_id', how='outer')
  df_merge['fraud_probabilities'].fillna(df_merge['fraud_probabilities'].mean(), inplace=True)

  return list(df_merge['fraud_probabilities'])

def albumincontradictionRule(df,dummydf):
  """
    "RULE: “No-Equipment/Entered-Data Contradiction” for Urine Test Albumin (No Equipment and then “Present/Absent for another patient)"
    
    Step 1. Processes the data in the given timeframe using the rule
    Step 2. Finds the percentages of non-diligence readings per ANM accoriding to the rule
    Step 3. Maps the percentages to non-diligence probabilities using KDE and outputs it

    Parameters
    ----------
    df : pandas DataFrame
        Dataframe to be processed
    dummydf : pandas DataFrame
        Dataframe with all the ANMs

    Returns
    -------
    list
        a list of non-diligence probabilities of ANMs
    """
  # Step 1
  grouped = df.groupby(['sub_center_id', 'camp_id','cluster_date'])
  bp = grouped.agg(
    Num_patients = pd.NamedAgg(column = 'ANC_Mother Id', aggfunc= lambda x: len(list(x))),
    Num_checked_patients = pd.NamedAgg(column = 'albumin', aggfunc= lambda x: len(list(x.dropna()))),
    albumin = pd.NamedAgg(column = 'albumin', aggfunc= lambda x: list(x.dropna()))   
  ).reset_index()

  # Step 2
  noEquipCount = []
  totalRealCount = []
  for k in range(len(bp)):
    noEquip = 0
    totReal = 0
    for i in range(len(bp.loc[k,'albumin'])):
      if (bp.loc[k,'albumin'][i]=='4'):
        noEquip += 1
      elif (bp.loc[k,'albumin'][i]=='True') or (bp.loc[k,'albumin'][i]=='False'):
        totReal += 1
    
    noEquipCount.append(noEquip)
    totalRealCount.append(totReal)

  bp = bp.assign(noEquipCount=noEquipCount)
  bp = bp.assign(totalRealCount=totalRealCount)
  bp['totalRealCount'].fillna(0, inplace=True)
  bp['suspicious_percentage'] = bp.apply(lambda row: row['noEquipCount']/row['Num_checked_patients']  if row['totalRealCount'] != 0 else 0, axis=1)

  anmgrouped = bp.groupby(['sub_center_id'])
  bpanm = anmgrouped.agg(
    Num_camps = pd.NamedAgg(column = 'suspicious_percentage', aggfunc= lambda x: len(list(x))),
    Tot_num_patients = pd.NamedAgg(column = 'Num_patients', aggfunc= sum),
    suspicious_percentage = pd.NamedAgg(column = 'suspicious_percentage', aggfunc= lambda x: list(x)),
    dates = pd.NamedAgg(column = 'cluster_date', aggfunc= lambda x: list(x))
  ).reset_index()

  # Step 3
  fraud_probabilities = []
  for i in range(len(bpanm)):
    fraud_prob = []
    for j in range(len(bpanm.loc[i,'suspicious_percentage'])):
      x = bpanm.loc[i,'suspicious_percentage'][j]
      #prob = func_get_prob_mass_trans(alb_contra_fit,0,x)
      fraud_prob.append(x)
    fraud_probabilities.append(np.mean(fraud_prob))

  fraud_probabilities = pd.Series(fraud_probabilities)
  bpanm = bpanm.assign(fraud_probabilities=fraud_probabilities.values)

  # merge with dummy dataframe
  df_merge = pd.merge(dummydf, bpanm, on='sub_center_id', how='outer')
  df_merge['fraud_probabilities'].fillna(df_merge['fraud_probabilities'].mean(), inplace=True)

  return list(df_merge['fraud_probabilities'])

# if > 80% of HIV Status = Not Done
def getHIVnotdoneRule(df,dummydf):
  """
    "RULE: if > 80% of HIV Status = Not Done"
    
    Step 1. Processes the data in the given timeframe using the rule
    Step 2. Finds the percentages of non-diligence readings per ANM accoriding to the rule
    Step 3. Maps the percentages to non-diligence probabilities using KDE and outputs it

    Parameters
    ----------
    df : pandas DataFrame
        Dataframe to be processed
    dummydf : pandas DataFrame
        Dataframe with all the ANMs

    Returns
    -------
    list
        a list of non-diligence probabilities of ANMs
    """
  # Step 1
  dfmod = df.loc[df['ANC_Mother Id'].notnull(), ['sub_center_id', 'camp_id', 'ANC_Mother Id', 'ANC_Hiv Test Result']]

  # Step 2
  bl = dfmod.groupby(['sub_center_id']).agg({'ANC_Mother Id': lambda x: len(x),
                                                        'ANC_Hiv Test Result': lambda x: len(x) - len(x.dropna())}).reset_index()
  bl['blankpercentage'] = bl.apply(lambda row: row['ANC_Hiv Test Result']/row['ANC_Mother Id'] , axis=1)

  # Step 3
  fraud_probabilities = []
  for i in range(len(bl)):
    x = float(bl.loc[i,'blankpercentage'])
    #prob = np.asarray(func_get_prob_mass_trans(hiv_fit,0,x))
    fraud_probabilities.append(x)

  fraud_probabilities = pd.Series(fraud_probabilities)
  bl = bl.assign(fraud_probabilities=fraud_probabilities)

  # merge with dummy dataframe
  df_merge = pd.merge(dummydf, bl, on='sub_center_id', how='outer')
  df_merge['fraud_probabilities'].fillna(np.mean(df_merge['fraud_probabilities']), inplace=True)

  return list(df_merge['fraud_probabilities'])

# if > 50% of Blood Type = Not Done
def getBloodGnotdoneRule(df,dummydf):
  """
    "RULE: if > 50% of Blood Type = Not Done"
    
    Step 1. Processes the data in the given timeframe using the rule
    Step 2. Finds the percentages of non-diligence readings per ANM accoriding to the rule
    Step 3. Maps the percentages to non-diligence probabilities using KDE and outputs it

    Parameters
    ----------
    df : pandas DataFrame
        Dataframe to be processed
    dummydf : pandas DataFrame
        Dataframe with all the ANMs

    Returns
    -------
    list
        a list of non-diligence probabilities of ANMs
    """
  # Step 1
  dfmod = df.loc[df['ANC_Mother Id'].notnull(), ['sub_center_id', 'camp_id', 'ANC_Mother Id', 'ANC_Blood Group']]

  # Step 2
  bl = dfmod.groupby(['sub_center_id']).agg({'ANC_Mother Id': lambda x: len(x),
                                                        'ANC_Blood Group': lambda x: len(x) - len(x.dropna())}).reset_index()
  bl['blankpercentage'] = bl.apply(lambda row: row['ANC_Blood Group']/row['ANC_Mother Id'] , axis=1)

  # Step 3
  fraud_probabilities = []
  for i in range(len(bl)):
    x = float(bl.loc[i,'blankpercentage'])
    #prob = np.asarray(func_get_prob_mass_trans(bloodg_fit,0,x))
    fraud_probabilities.append(x)

  fraud_probabilities = pd.Series(fraud_probabilities)
  bl = bl.assign(fraud_probabilities=fraud_probabilities.values)

  # merge with dummy dataframe
  df_merge = pd.merge(dummydf, bl, on='sub_center_id', how='outer')
  df_merge['fraud_probabilities'].fillna(df_merge['fraud_probabilities'].mean(), inplace=True)

  return list(df_merge['fraud_probabilities'])

# if > 85% of VDRL Status = Not Done
def getVDRLnotdoneRule(df,dummydf):
  """
    "RULE: if > 85% of VDRL Status = Not Done"
    
    Step 1. Processes the data in the given timeframe using the rule
    Step 2. Finds the percentages of non-diligence readings per ANM accoriding to the rule
    Step 3. Maps the percentages to non-diligence probabilities using KDE and outputs it

    Parameters
    ----------
    df : pandas DataFrame
        Dataframe to be processed
    dummydf : pandas DataFrame
        Dataframe with all the ANMs

    Returns
    -------
    list
        a list of non-diligence probabilities of ANMs
    """
  # Step 1
  dfmod = df.loc[df['ANC_Mother Id'].notnull(), ['sub_center_id', 'camp_id', 'ANC_Mother Id', 'ANC_Vdrl Test Result']]

  # Step 2
  bl = dfmod.groupby(['sub_center_id']).agg({'ANC_Mother Id': lambda x: len(x),
                                                        'ANC_Vdrl Test Result': lambda x: len(x) - len(x.dropna())}).reset_index()
  bl['blankpercentage'] = bl.apply(lambda row: row['ANC_Vdrl Test Result']/row['ANC_Mother Id'] , axis=1)

  # Step 3
  fraud_probabilities = []
  for i in range(len(bl)):
    x = float(bl.loc[i,'blankpercentage'])
    #prob = np.asarray(func_get_prob_mass_trans(vdrl_fit,0,x))
    fraud_probabilities.append(x)

  fraud_probabilities = pd.Series(fraud_probabilities)
  bl = bl.assign(fraud_probabilities=fraud_probabilities.values)

  # merge with dummy dataframe
  df_merge = pd.merge(dummydf, bl, on='sub_center_id', how='outer')
  df_merge['fraud_probabilities'].fillna(df_merge['fraud_probabilities'].mean(), inplace=True)

  return list(df_merge['fraud_probabilities'])

# More than 3 back-to-back visits with test denial showing “didn’t know how to do test” - particularly for fundal height
def getFundalHeightRule(df,dummydf):
  """
    "RULE: More than 3 back-to-back visits with test denial showing “didn’t know how to do test” - particularly for fundal height"
    
    Step 1. Processes the data in the given timeframe using the rule
    Step 2. Finds the percentages of non-diligence readings per ANM accoriding to the rule
    Step 3. Maps the percentages to non-diligence probabilities using KDE and outputs it

    Parameters
    ----------
    df : pandas DataFrame
        Dataframe to be processed
    dummydf : pandas DataFrame
        Dataframe with all the ANMs

    Returns
    -------
    list
        a list of non-diligence probabilities of ANMs
    """
  # Step 1
  dfmod = df.loc[df['ANC_Mother Id'].notnull(), ['sub_center_id', 'camp_id','ANC_Mother Id',
    'ANC_ANC1 Fundal Height', 'ANC_ANC2 Fundal Height', 'ANC_ANC3 Fundal Height', 'ANC_ANC4 Fundal Height']]

  # Step 2
  dfmod['fraud1'] = dfmod.apply(lambda row: 1 if (row['ANC_ANC1 Fundal Height']==2 and row['ANC_ANC2 Fundal Height']==2 and row['ANC_ANC3 Fundal Height']==2) else 0, axis=1)
  dfmod['fraud2'] = dfmod.apply(lambda row: 1 if (row['ANC_ANC2 Fundal Height']==2 and row['ANC_ANC3 Fundal Height']==2 and row['ANC_ANC4 Fundal Height']==2) else 0, axis=1)
  dfmod['fraudFundalH'] = dfmod.apply(lambda row: 1 if (row['fraud1']==1 or row['fraud2']==1) else 0, axis=1)
  dfanm3 = dfmod.groupby(['sub_center_id']).agg({'ANC_Mother Id': lambda x: len(x),
                                                'fraudFundalH': lambda x: tuple(x)}).reset_index()
  dfanm3['suspicious_percentage'] = dfanm3.apply(
        lambda row: row['fraudFundalH'].count(1) /len(row['fraudFundalH'])  if len(row['fraudFundalH'])!=0 else 0, axis=1)
    
  # Step 3
  fraud_probabilities = []
  for i in range(len(dfanm3)):
    x = float(dfanm3.loc[i,'suspicious_percentage'])
    #prob = np.asarray(func_get_prob_mass_trans(fundalh_fit,0,x))
    fraud_probabilities.append(x)

  fraud_probabilities = pd.Series(fraud_probabilities)
  dfanm3 = dfanm3.assign(fraud_probabilities=fraud_probabilities.values)

  # merge with dummy dataframe
  df_merge = pd.merge(dummydf, dfanm3, on='sub_center_id', how='outer')
  df_merge['fraud_probabilities'].fillna(df_merge['fraud_probabilities'].mean(), inplace=True)

  return list(df_merge['fraud_probabilities'])

#Greater than 90% of mothers with child less than 6 months have EBF = True
def getebfRule(df,dummydf):
  """
    "RULE: Greater than 90% of mothers with child less than 6 months have EBF = True"
    
    Step 1. Processes the data in the given timeframe using the rule
    Step 2. Finds the percentages of non-diligence readings per ANM accoriding to the rule
    Step 3. Maps the percentages to non-diligence probabilities using KDE and outputs it

    Parameters
    ----------
    df : pandas DataFrame
        Dataframe to be processed
    dummydf : pandas DataFrame
        Dataframe with all the ANMs

    Returns
    -------
    list
        a list of non-diligence probabilities of ANMs
    """
  # Step 1
  dfmod = df.loc[df['Delivery_Delivery Outcome ID'].notnull(), ['sub_center_id', 'camp_id','Delivery_Delivery Outcome ID', 'Delivery_Breastfeeding from birth']]
  
  # Step 2
  grouped = dfmod.groupby(['sub_center_id'])
  ebf = grouped.agg(
    val = pd.NamedAgg(column='Delivery_Breastfeeding from birth', aggfunc= lambda x: tuple(x.dropna())), 
    val_count = pd.NamedAgg(column='Delivery_Breastfeeding from birth', aggfunc=lambda x: len(tuple(x.dropna()))),
    true_count = pd.NamedAgg(column='Delivery_Breastfeeding from birth', aggfunc=lambda x: tuple(x.dropna()).count(True))
  ).reset_index()
  ebf['true_percentage'] = ebf.apply(lambda row: row['true_count']/row['val_count']  if row['val_count']!=0 else 0, axis=1)

  # Step 3
  fraud_probabilities = []
  for i in range(len(ebf)):
    x = float(ebf.loc[i,'true_percentage'])
    #prob = np.asarray(func_get_prob_mass_trans(ebf_fit,0,x))
    fraud_probabilities.append(x)

  fraud_probabilities = pd.Series(fraud_probabilities)
  ebf = ebf.assign(fraud_probabilities=fraud_probabilities.values)

  # merge with dummy dataframe
  df_merge = pd.merge(dummydf, ebf, on='sub_center_id', how='outer')
  df_merge['fraud_probabilities'].fillna(df_merge['fraud_probabilities'].mean(), inplace=True)

  return list(df_merge['fraud_probabilities'])

# Proportion of child death <2%
def deathRule(dfr,dummydf):
  """
    "RULE: Proportion of child death <2%"
    
    Step 1. Processes the data in the given timeframe using the rule
    Step 2. Finds the percentages of non-diligence readings per ANM accoriding to the rule
    Step 3. Maps the percentages to non-diligence probabilities using KDE and outputs it

    Parameters
    ----------
    df : pandas DataFrame
        Dataframe to be processed
    dummydf : pandas DataFrame
        Dataframe with all the ANMs

    Returns
    -------
    list
        a list of non-diligence probabilities of ANMs
    """
  # Step 1
  dfmod = dfr.loc[dfr['ANC_Mother Id'].notnull(), ['sub_center_id', 'camp_id','ANC_Mother Id','Delivery_Pregnancy Out Come']]
  
  # Step 2
  grouped = dfmod.groupby(['sub_center_id'])
  dr = grouped.agg(
    live_birth_count = pd.NamedAgg(column='Delivery_Pregnancy Out Come', aggfunc=lambda x: tuple(x.dropna()).count(1)),
    still_birth_count = pd.NamedAgg(column='Delivery_Pregnancy Out Come', aggfunc=lambda x: tuple(x.dropna()).count(2)),
    abortion_count = pd.NamedAgg(column='Delivery_Pregnancy Out Come', aggfunc=lambda x: tuple(x.dropna()).count(3))
  ).reset_index()

  dr['death_percentage'] = dr.apply(lambda row: row['still_birth_count']/(row['live_birth_count']+row['still_birth_count'])  if (row['live_birth_count']+row['still_birth_count'])!=0 else 0, axis=1)

  # Step 3
  fraud_probabilities = []
  for i in range(len(dr)):
    x = float(dr.loc[i,'death_percentage'])
    #prob = np.asarray(func_get_prob_mass_trans(death_fit,x,100))
    fraud_probabilities.append(x)

  fraud_probabilities = pd.Series(fraud_probabilities)
  dr = dr.assign(fraud_probabilities=fraud_probabilities.values)

  # merge with dummy dataframe
  df_merge = pd.merge(dummydf, dr, on='sub_center_id', how='outer')
  df_merge['fraud_probabilities'].fillna(df_merge['fraud_probabilities'].mean(), inplace=True)

  return list(df_merge['fraud_probabilities'])

"""Merge algo functions to cluster the dates of a camp together, since the reported date of medical records may vary from the actual camp date."""

# runs the merge algo for all camps in all ANMs and return the date clusters with the merged dates for each ANM
def mergeAlgo(df):
  """
    runs the merge algo for all camps in all ANMs and return the date clusters with the merged dates for each ANM

    Step 1. finds clusters for values less than the treshold by searching the given range
    Step 2. expands the cluster given number of times by searching given range into past
    Step 3. handle if there is a value larger than threshold closer to considered date than to the date with highest value in the cluster

    Parameters
    ----------
    df : pandas DataFrame
        Dataframe to be processed (camp-datewise preprocessed input)

    Returns
    -------
    dict_campClusterDates : dictionary
        a dictionary with dates to be clustered together for each camp_id
    dict_campUnifiedDate : dictionary
        a dictionary with the cluster dates to be represented after merging few dates together for each camp_id
    """

  # set parameters
  threshold = 3             #threshold for the dates to be clustered if number of patients is less than or equal to
  searchRange = 7           #number of days the searching is done 
  expandClusterTimes = 2    #how may times the cluster is expanded in scenario 1
  
  # get all ANM and camps for clustering
  testANM, testcamp = getANMCampIds(df)

  dict_campClusterDates = {}
  dict_campUnifiedDate = {}

  originalDatesDF = getOriginalGroupedByDate(df)
  originalDatesDF2 = afterSystemOnline(originalDatesDF)

  sortedOriginalDates = originalDatesDF2.sort_values(by=['date'], inplace=False, ascending=False)
  print("Date range: ", sortedOriginalDates['date'].iloc[0],sortedOriginalDates['date'].iloc[-1])

  # dummy dataframe
  dates = pd.date_range(start=sortedOriginalDates['date'].iloc[-1], end=sortedOriginalDates['date'].iloc[0], freq='D')
  dummydates = pd.DataFrame({ 'date': dates, 'Val': np.random.randn(len(dates)) })

  for i in range(len(testANM)):
    for campid in testcamp[i]:
      #print("CAMP: ", campid)
      campClusterDates = []
      campUnifiedDate = []
      campdf = sortedOriginalDates[(sortedOriginalDates['sub_center_id']==testANM[i]) & (sortedOriginalDates['camp_id']==campid)].reset_index()

      # merge with dummy dataframe
      df_merge = pd.merge(dummydates, campdf, on='date', how='outer')
      df_merge['Num_patients'].fillna(0, inplace=True)

      datesOriginal = df_merge['date']
      patientsOriginal = df_merge['Num_patients']
      Num_dates = len(df_merge['date'])
      datesMerged = np.array(df_merge['date'], dtype=object)
      patientsMerged = np.array(df_merge['Num_patients'])

      # finds clusters for values less than the treshold
      for d in range(0, Num_dates):
        if((patientsMerged[d] <= threshold) and (patientsMerged[d] > 0)):
          clusterDates = [datesMerged[d]]
          clusterPatients = [patientsMerged[d]]
          clusterIdx = [d]

          # search for non-zero values in +/- searchRange (end value inclusive)
          for r in range(1,searchRange+1):
            # search searchRange days to future
            if ( (d-r) > 0):
              if (patientsMerged[d-r] > 0):
                clusterDates.append(datesMerged[d-r])
                clusterPatients.append(patientsMerged[d-r])
                clusterIdx.append(d-r)

            # search searchRange days to the past
            if ( (d+r) < Num_dates):
              if (patientsMerged[d+r] > 0):
                clusterDates.append(datesMerged[d+r])
                clusterPatients.append(patientsMerged[d+r])
                clusterIdx.append(d+r)
  
          # scenario 1
          clusterEarliestDayIdx = np.max(clusterIdx) 
          if (clusterEarliestDayIdx > d):
            flag = False
            # expands the cluster given number of times by searching given range into past
            for t in range(expandClusterTimes):
              # search searchRange days to the past
              for r in range(1,searchRange+1):
                if ( (clusterEarliestDayIdx + r) < Num_dates):
                  if (patientsMerged[clusterEarliestDayIdx + r] > threshold):
                    # stops cluster expansion if a value greater than threshold is found
                    flag = True
                    break
                  elif (patientsMerged[clusterEarliestDayIdx +r ] > 0):
                    # expands cluster if a non-zero value less than threshold is found
                    clusterDates.append(datesMerged[clusterEarliestDayIdx + r])
                    clusterPatients.append(patientsMerged[clusterEarliestDayIdx + r])
                    clusterIdx.append(clusterEarliestDayIdx + r)
              if(flag):
                break
              clusterEarliestDayIdx = np.max(clusterIdx)


          #scenario 2
          clusterPatients = np.array(clusterPatients)
          clusterDates = np.array(clusterDates)
          clusterIdx = np.array(clusterIdx)
          belowThresholdIdx = clusterIdx[clusterPatients <= threshold]
          # select the leftmost below threshold value idx
          pastBelowThresholdIdx = np.max(belowThresholdIdx)
          max_idx = clusterIdx[np.argmax(clusterPatients)]
          if (max_idx < pastBelowThresholdIdx):
          # search searchRange-1 days backwards for values above threshold
            for r in range(1,searchRange):
              if ( (pastBelowThresholdIdx + r) < Num_dates):
                if (patientsMerged[pastBelowThresholdIdx + r] > threshold):
                  if (r >= (pastBelowThresholdIdx - max_idx)):
                    break
                  else:
                    # if there is a value larger than threshold closer to considered date than to the date with highest value in the cluster
                    # remove all leftmost values from considered date 
                    remove_val = pastBelowThresholdIdx + r
                    clusterIdx = clusterIdx[np.where(clusterIdx < remove_val)]
                    clusterPatients = clusterPatients[np.where(clusterIdx < remove_val)]
                    clusterDates = clusterDates[np.where(clusterIdx < remove_val)]
                         

          if(len(clusterPatients) == 1):
            # no non zero values found in searchRange
            continue
          else:
            # if cluster is found, merge them together and assign to the date with highest patients
            max_idx = clusterIdx[np.argmax(clusterPatients)]
            clusterSum = np.sum(clusterPatients)
            
            for c in clusterIdx:
              if(c==max_idx):
                patientsMerged[c] = clusterSum
              else:
                patientsMerged[c] = 0

          campClusterDates.append(clusterDates)
          campUnifiedDate.append(datesMerged[max_idx])
          #print(clusterDates)

      dict_campClusterDates[campid] = campClusterDates
      dict_campUnifiedDate[campid] = campUnifiedDate
      
  return dict_campClusterDates, dict_campUnifiedDate


# gets the cluster dates and merged dates for each camp and label all the records accordingly
def addClusters(dfm,df):
  """
    Gets the cluster dates and merged dates for each camp by running the merge algo
    and label all the records accordingly with new merged dates

    Parameters
    ----------
    dfm : pandas DataFrame
        Dataframe to be processed (camp-datewise preprocessed input)
    df : pandas DataFrame
        Dataframe to be processed (original)

    Returns
    -------
    dfm : pandas DataFrame
        Dataframe with the added cluster dates
    """

  dict_campClusterDates, dict_campUnifiedDate = mergeAlgo(df)

  dfm['cluster_date'] = np.nan
  campIds = getAllCampIds(df)

  for i in range(len(campIds)):
    campClusterDates = dict_campClusterDates[campIds[i]]
    campUnifiedDate = dict_campUnifiedDate[campIds[i]]

    for j in range(len(campUnifiedDate)):
      for k in range(len(campClusterDates[j])):
        mask = (dfm['date'] == campClusterDates[j][k]) & (dfm['camp_id'] == campIds[i])
        dfm.loc[mask, 'cluster_date'] = campUnifiedDate[j]

  dfm['cluster_date'] = pd.to_datetime(dfm['cluster_date'])
  dfm['cluster_date'].fillna(dfm['date'],inplace=True)

  return dfm

"""Helper functions for pre-processing data"""

def read_data():
  """
    Reads CSV

    Returns
    -------
    df : pandas DataFrame
        Dataframe with unique mothers
    dfr : pandas DataFrame
        Dataframe with twins (original)
    """
  dfr = pd.read_excel('data/Master_Dataset_KB-Google.xlsx')
  df = dfr.drop_duplicates(subset=['ANC_Mother Id'], keep='first', inplace=False).reset_index()
  return df, dfr

def get_campdate_level_df(df):
  columns = ['sub_center_id', 'camp_id', 'ANC_Mother Id','ANC_ANC1 Date','ANC_ANC2 Date','ANC_ANC3 Date','ANC_ANC4 Date', 
             'ANC_ANC1 BP Sys', 'ANC_ANC1 BP Dia', 'ANC_ANC2 BP Sys', 'ANC_ANC2 BP Dia', 'ANC_ANC3 BP Sys', 
             'ANC_ANC3 BP Dia', 'ANC_ANC4 BP Sys', 'ANC_ANC4 BP Dia',
             'ANC_ANC1 Urine Albumin', 'ANC_ANC2 Urine Albumin', 'ANC_ANC3 Urine Albumin', 'ANC_ANC4 Urine Albumin',
             'ANC_ANC1 Urine Sugar','ANC_ANC2 Urine Sugar','ANC_ANC3 Urine Sugar','ANC_ANC4 Urine Sugar',
             'ANC_ANC1 HB','ANC_ANC2 HB','ANC_ANC3 HB','ANC_ANC4 HB',
             'ANC_ANC1 Weight', 'ANC_ANC2 Weight', 'ANC_ANC3 Weight', 'ANC_ANC4 Weight',
             'ANC_ANC1 Blood Sugar','ANC_ANC2 Blood Sugar','ANC_ANC3 Blood Sugar','ANC_ANC4 Blood Sugar',
             'ANC_ANC1 Fetal Heart Rate','ANC_ANC2 Fetal Heart Rate','ANC_ANC3 Fetal Heart Rate','ANC_ANC4 Fetal Heart Rate']
  dfmod = df.loc[df['ANC_Mother Id'].notnull(), columns]

  newcolumns = ['date','BPsys', 'BPdia','urinesugar','albumin','hb','weight','blood_sugar','fetalhr']
  oldcolumns = [['ANC_ANC1 Date','ANC_ANC2 Date','ANC_ANC3 Date','ANC_ANC4 Date'],
                ['ANC_ANC1 BP Sys', 'ANC_ANC2 BP Sys', 'ANC_ANC3 BP Sys', 'ANC_ANC4 BP Sys'],
                ['ANC_ANC1 BP Dia', 'ANC_ANC2 BP Dia', 'ANC_ANC3 BP Dia', 'ANC_ANC4 BP Dia'],
                ['ANC_ANC1 Urine Sugar','ANC_ANC2 Urine Sugar','ANC_ANC3 Urine Sugar','ANC_ANC4 Urine Sugar'],
                ['ANC_ANC1 Urine Albumin', 'ANC_ANC2 Urine Albumin', 'ANC_ANC3 Urine Albumin', 'ANC_ANC4 Urine Albumin'],
                ['ANC_ANC1 HB','ANC_ANC2 HB','ANC_ANC3 HB','ANC_ANC4 HB'],
                ['ANC_ANC1 Weight', 'ANC_ANC2 Weight', 'ANC_ANC3 Weight', 'ANC_ANC4 Weight'],
                ['ANC_ANC1 Blood Sugar','ANC_ANC2 Blood Sugar','ANC_ANC3 Blood Sugar','ANC_ANC4 Blood Sugar'],
                ['ANC_ANC1 Fetal Heart Rate','ANC_ANC2 Fetal Heart Rate','ANC_ANC3 Fetal Heart Rate','ANC_ANC4 Fetal Heart Rate']]

  df2 = arrangeDate(dfmod,newcolumns,oldcolumns)
  df3 = afterSystemOnline(df2)
  df4 = addClusters(df3,df)

  return df4

# Rearranges the data: to concat 4 ANC dates and medical data relevant to 4 dates
def arrangeDate(df, newcolumns, oldcolumns):
  columns= ['sub_center_id','camp_id','ANC_Mother Id']
  columns = np.concatenate((columns, newcolumns))
  df2 = pd.DataFrame(columns=columns)
  df2['sub_center_id'] = pd.concat([df['sub_center_id'], df['sub_center_id'], df['sub_center_id'], df['sub_center_id']], ignore_index=True)
  df2['camp_id'] = pd.concat([df['camp_id'], df['camp_id'], df['camp_id'], df['camp_id']], ignore_index=True)
  df2['ANC_Mother Id'] = pd.concat([df['ANC_Mother Id'], df['ANC_Mother Id'], df['ANC_Mother Id'], df['ANC_Mother Id']], ignore_index=True)
  for i in range(len(newcolumns)):
    df2[newcolumns[i]] = pd.concat([df[oldcolumns[i][j]] for j in range(4)],ignore_index=True)

  return df2

def afterSystemOnline(df3, column='date'):
  df3[column] = pd.to_datetime(df3[column])
  df4 = df3[(df3[column] > '2017-02-08')].reset_index()
  return df4

# get all camp ids
def getAllCampIds(df):
   return list(set(df['camp_id']))

# get all camp ids and ANM ids
def getANMCampIds(df):
  df2 = df.groupby(['sub_center_id']).agg(
    camp_id = pd.NamedAgg(column = 'camp_id', aggfunc= lambda x: list(set(x)))
  ).reset_index()
  
  return list(df2['sub_center_id']), list(df2['camp_id'])

def getOriginalGroupedByDate(df):
  columns = ['sub_center_id', 'camp_id', 'ANC_Mother Id','ANC_ANC1 Date','ANC_ANC2 Date','ANC_ANC3 Date','ANC_ANC4 Date']
  dfmod = df.loc[df['ANC_Mother Id'].notnull(), columns]
  newcolumns = ['date']
  oldcolumns = [['ANC_ANC1 Date','ANC_ANC2 Date','ANC_ANC3 Date','ANC_ANC4 Date']]
  
  df2 = arrangeDate(dfmod,newcolumns,oldcolumns)
  
  grouped = df2.groupby(['sub_center_id','camp_id','date'])
  df3 = grouped.agg(
    Num_patients = pd.NamedAgg(column = 'ANC_Mother Id', aggfunc= lambda x: len(tuple(x)))
  ).reset_index()

  return df3

def get_sliding_dates(window_size_weeks):
  online_day = datetime.date(2017, 2, 8)
  data_end_day = datetime.date(2020, 3, 19)
  slide_start_day = online_day + relativedelta(months=+6)

  sliding_dates = [[window_size_weeks]]
  window_dates = []
  last_day = slide_start_day

  while (last_day < data_end_day):
    window_dates.append(last_day)
    last_day += relativedelta(weeks=+window_size_weeks)
    
  sliding_dates.append(window_dates)

  return sliding_dates

def get_window_df(df, start_date, end_date, column='cluster_date'):
  df[column] = pd.to_datetime(df[column])
  mask = (df[column] > pd.to_datetime(start_date)) & (df[column] <= pd.to_datetime(end_date))
  new_df = df.loc[mask]

  return new_df

def get_dummy_anm_df(df):
  allANM, _ = getANMCampIds(df)
  dummydf = pd.DataFrame({ 'sub_center_id': allANM, 'Val': np.random.randn(len(allANM)) })
  return dummydf

# Gets the fraud probabilities for each rule by running a sliding window
def get_fraud_probabilities(window_size_weeks = 4):
  
  camp_level_rules = [getBPsdRuleData, getModfUrineRule, getBPhyperRule, getHBrule,
    getHB7rule, bpcontradictionRule, weightcontradictionRule, hbcontradictionRule, 
    bloodsugarcontradictionRule, fetalhrcontradictionRule, urinesugarcontradictionRule, albumincontradictionRule]
  patient_level_rules = [getHIVnotdoneRule, getBloodGnotdoneRule, getVDRLnotdoneRule, getFundalHeightRule, getebfRule]
  patient_dates_columns = ['ANC_EDD','ANC_EDD','ANC_EDD','ANC_EDD', 'Delivery_Delivery Date']

  df, dftwins = read_data()
  all_campdate_df = get_campdate_level_df(df)
  sliding_dates = get_sliding_dates(window_size_weeks)
  dummydf = get_dummy_anm_df(df)
  print(sliding_dates)

  # 35 windows * rules * ~85 probabilities
  window_probabilities = []
  for i in range(len(sliding_dates[1])):
  #for i in range(2):
    rule_probabilities = []
    end_date = sliding_dates[1][i]

    # camp date level rules
    start_date = end_date + relativedelta(weeks=-window_size_weeks)
    campdate_df = get_window_df(all_campdate_df, start_date, end_date)
    for rule in camp_level_rules:
      probabilities = rule(campdate_df, dummydf)
      rule_probabilities.append(probabilities)

    # for patient level rules
    for j in range(len(patient_level_rules)):
      p_start_date = end_date + relativedelta(months=-6)
      patient_df = get_window_df(df, p_start_date, end_date, column=patient_dates_columns[j])
      probabilities = patient_level_rules[j](patient_df,dummydf)
      rule_probabilities.append(probabilities)

    # for death rule
    d_start_date = end_date + relativedelta(months=-6)
    patient_df = get_window_df(dftwins, d_start_date, end_date, column='ANC_Date Of Outcome')
    probabilities = deathRule(patient_df,dummydf)
    rule_probabilities.append(probabilities)

    window_probabilities.append(rule_probabilities)
  
  window_probabilities = np.array(window_probabilities)
  print(window_probabilities.shape)

  return window_probabilities

# fraud probability vector history for each point
def getHistory(fraud_prob_vectors, num_months=6, num_anm=85):
  ign = num_anm*num_months
  num_points = len(fraud_prob_vectors) - ign
  fraud_vect_history = []
  for i in range(num_points):
    idx = ign + i
    h = []
    for j in range(1,num_months+1):
      k = num_months + 1 - j
      vect = fraud_prob_vectors[idx-k*num_anm]
      h.append(vect)
    fraud_vect_history.append(h)
  
  fraud_vect_history = np.asarray(fraud_vect_history)
  print(fraud_vect_history.shape)

  return fraud_vect_history

# Gets the ANM's other features
def get_features(window_size_weeks = 4):

  df, _ = read_data()
  all_campdate_df = get_campdate_level_df(df)
  sliding_dates = get_sliding_dates(window_size_weeks)
  dummydf = get_dummy_anm_df(df)
  print(sliding_dates)

  # 35 windows * 85 
  features = []
  t1 = []
  for i in range(len(sliding_dates[1])):
  #for i in range(2):

    end_date = sliding_dates[1][i]
    start_date = end_date + relativedelta(weeks=-window_size_weeks)
    campdate_df = get_window_df(all_campdate_df, start_date, end_date)

    window_features, t = get_window_features(campdate_df, dummydf)
    t1.append(t)
    features.append(window_features)

  features = np.asarray(features)
  print(features.shape)
  print(np.sum(t1))

  return features

def get_window_features(campdate_df, dummydf):

  grouped = campdate_df.groupby(['sub_center_id', 'camp_id'])
  bp = grouped.agg(
    Tot_num_patients = pd.NamedAgg(column = 'ANC_Mother Id', aggfunc= lambda x: len(list(x))),
    Num_camps = pd.NamedAgg(column = 'cluster_date', aggfunc= lambda x: len(set(x))),
    dates = pd.NamedAgg(column = 'cluster_date', aggfunc= lambda x: list(set(x)))
  ).reset_index()

  anmgrouped = bp.groupby(['sub_center_id'])
  bpanm = anmgrouped.agg(
    Tot_num_patients = pd.NamedAgg(column = 'Tot_num_patients', aggfunc= sum),
    Num_camps = pd.NamedAgg(column = 'Num_camps', aggfunc= sum),
    Num_locations = pd.NamedAgg(column = 'camp_id', aggfunc= lambda x: len(list(x))),
    dates = pd.NamedAgg(column = 'dates', aggfunc= lambda x: list(x))
  ).reset_index()

  dates_diff = np.array([])
  for i in range(len(bpanm)):    
    num_camps = bpanm.loc[i,'Num_camps']
    if num_camps<2:
      dates_diff = np.append(dates_diff,28)
    else:
      anm_dates = bpanm.loc[i,'dates']
      flat_list = [item for sublist in anm_dates for item in sublist]
      flat_list.sort()
      anm_diff = np.array([])
      for j in range(1,len(flat_list)):
        diff = (flat_list[j] - flat_list[j-1]).days
        anm_diff = np.append(anm_diff,diff)
      dates_diff = np.append(dates_diff, np.mean(anm_diff))

  dates_diff = pd.Series(dates_diff)
  bpanm = bpanm.assign(dates_diff=dates_diff.values)

  # merge with dummy dataframe
  df_merge = pd.merge(dummydf, bpanm, on='sub_center_id', how='outer')
  
  t = df_merge['Tot_num_patients'].isna().sum()
  df_merge['Tot_num_patients'].fillna(0, inplace=True)
  df_merge['Num_camps'].fillna(0, inplace=True)
  df_merge['dates_diff'].fillna(0, inplace=True)
  df_merge['Num_locations'].fillna(0, inplace=True)

  df_merge.to_csv('test2.csv')

  window_features = np.asarray([np.asarray(df_merge['Tot_num_patients']), np.asarray(df_merge['Num_camps']), np.asarray(df_merge['dates_diff']), np.asarray(df_merge['Num_locations'])])

  return np.transpose(window_features), t

# 35 windows * 18 rules * 85 ANM
fraud_probabilities = get_fraud_probabilities()
fraud_prob = fraud_probabilities.swapaxes(1,2)
shape2 = fraud_prob.shape
test_fraud_prob = fraud_prob.reshape(shape2[0]*shape2[1],shape2[2])
print(test_fraud_prob.shape)

#history
fraud_vect_history = getHistory(test_fraud_prob)

# other features of the ANM
features = get_features()
fshape = features.shape
inp1 = features.reshape(fshape[0]*fshape[1],fshape[2])
valid_inp1 = inp1[85*6:]
print(valid_inp1.shape)

inp = [valid_inp1, fraud_vect_history]

# creates a mask to seperate training and testing sets
msk1 = np.tile([True], 19*85)
msk2 = np.tile([False], 10*85)
msk = np.append(msk1,msk2)
print("Size: ",len(msk))
print("Train set size: ", len(msk1))
print("Test set size: ", len(msk2))

inp1_train = valid_inp1[msk]
inp1_test = valid_inp1[~msk]
inp2_train = fraud_vect_history[msk]
inp2_test = fraud_vect_history[~msk]
x_train = [inp1_train, inp2_train]
x_test = [inp1_test, inp2_test]
print(inp1_train.shape, inp1_test.shape)


"""New data set"""

def read_data_new():
  dfr = pd.read_excel('data/kb_googlai_master_folder/Google_KB_Test_Dataset.xlsx')
  df = dfr.drop_duplicates(subset=['ANC_Mother Id'], keep='first', inplace=False).reset_index()

  return df, dfr

def get_sliding_dates_new(window_size_weeks):
  online_day = datetime.date(2020, 4, 30)
  data_end_day = datetime.date(2020, 8, 20)
  slide_start_day = online_day + relativedelta(weeks=+window_size_weeks)

  sliding_dates = [[window_size_weeks]]
  window_dates = []
  last_day = slide_start_day

  while (last_day <= data_end_day):
    window_dates.append(last_day)
    last_day += relativedelta(weeks=+window_size_weeks)
    
  sliding_dates.append(window_dates)

  return sliding_dates

def afterSystemOnline_new(df3, column='date'):
  df3[column] = pd.to_datetime(df3[column])
  df4 = df3[(df3[column] > '2020-04-30')].reset_index()
  return df4

def filter_new_data(df):
  df['ANC_ANC1 Date'] = pd.to_datetime(df['ANC_ANC1 Date'])
  df['ANC_ANC2 Date'] = pd.to_datetime(df['ANC_ANC2 Date'])
  df['ANC_ANC3 Date'] = pd.to_datetime(df['ANC_ANC3 Date'])
  df['ANC_ANC4 Date'] = pd.to_datetime(df['ANC_ANC4 Date'])
  filt = (df['ANC_ANC1 Date'] > '2020-04-30') | (df['ANC_ANC2 Date'] > '2020-04-30') | (df['ANC_ANC3 Date'] > '2020-04-30') | (df['ANC_ANC4 Date'] > '2020-04-30')
  filt_df = df[filt]
  return filt_df
  
def get_campdate_level_df_new(df):
  columns = ['sub_center_id', 'camp_id', 'ANC_Mother Id','ANC_ANC1 Date','ANC_ANC2 Date','ANC_ANC3 Date','ANC_ANC4 Date', 
             'ANC_ANC1 BP Sys', 'ANC_ANC1 BP Dia', 'ANC_ANC2 BP Sys', 'ANC_ANC2 BP Dia', 'ANC_ANC3 BP Sys', 
             'ANC_ANC3 BP Dia', 'ANC_ANC4 BP Sys', 'ANC_ANC4 BP Dia',
             'ANC_ANC1 Urine Albumin', 'ANC_ANC2 Urine Albumin', 'ANC_ANC3 Urine Albumin', 'ANC_ANC4 Urine Albumin',
             'ANC_ANC1 Urine Sugar','ANC_ANC2 Urine Sugar','ANC_ANC3 Urine Sugar','ANC_ANC4 Urine Sugar',
             'ANC_ANC1 HB','ANC_ANC2 HB','ANC_ANC3 HB','ANC_ANC4 HB',
             'ANC_ANC1 Weight', 'ANC_ANC2 Weight', 'ANC_ANC3 Weight', 'ANC_ANC4 Weight',
             'ANC_ANC1 Blood Sugar','ANC_ANC2 Blood Sugar','ANC_ANC3 Blood Sugar','ANC_ANC4 Blood Sugar',
             'ANC_ANC1 Fetal Heart Rate','ANC_ANC2 Fetal Heart Rate','ANC_ANC3 Fetal Heart Rate','ANC_ANC4 Fetal Heart Rate']
  dfmod = df.loc[df['ANC_Mother Id'].notnull(), columns]

  newcolumns = ['date','BPsys', 'BPdia','urinesugar','albumin','hb','weight','blood_sugar','fetalhr']
  oldcolumns = [['ANC_ANC1 Date','ANC_ANC2 Date','ANC_ANC3 Date','ANC_ANC4 Date'],
                ['ANC_ANC1 BP Sys', 'ANC_ANC2 BP Sys', 'ANC_ANC3 BP Sys', 'ANC_ANC4 BP Sys'],
                ['ANC_ANC1 BP Dia', 'ANC_ANC2 BP Dia', 'ANC_ANC3 BP Dia', 'ANC_ANC4 BP Dia'],
                ['ANC_ANC1 Urine Sugar','ANC_ANC2 Urine Sugar','ANC_ANC3 Urine Sugar','ANC_ANC4 Urine Sugar'],
                ['ANC_ANC1 Urine Albumin', 'ANC_ANC2 Urine Albumin', 'ANC_ANC3 Urine Albumin', 'ANC_ANC4 Urine Albumin'],
                ['ANC_ANC1 HB','ANC_ANC2 HB','ANC_ANC3 HB','ANC_ANC4 HB'],
                ['ANC_ANC1 Weight', 'ANC_ANC2 Weight', 'ANC_ANC3 Weight', 'ANC_ANC4 Weight'],
                ['ANC_ANC1 Blood Sugar','ANC_ANC2 Blood Sugar','ANC_ANC3 Blood Sugar','ANC_ANC4 Blood Sugar'],
                ['ANC_ANC1 Fetal Heart Rate','ANC_ANC2 Fetal Heart Rate','ANC_ANC3 Fetal Heart Rate','ANC_ANC4 Fetal Heart Rate']]

  df2 = arrangeDate(dfmod,newcolumns,oldcolumns)
  df3 = afterSystemOnline_new(df2)
  df4 = addClusters(df3,df)

  return df4

def get_one_hot(vect):
  out = np.zeros((vect.size, vect.max()+1))
  out[np.arange(vect.size),vect] = 1
  return out

def remove_zero(valid_in, in2, y):
  maskn = np.tile([True],len(valid_in))
  for i in range(len(valid_in)):
    if valid_in[i][0] == 0:
      maskn[i] = False

  in1_2 = valid_in[maskn]
  in2_2 = in2[maskn]
  y_2 = y[maskn]
  x_2 = [in1_2, in2_2]

  return x_2, y_2, maskn

def get_window_features(campdate_df, dummydf):

  grouped = campdate_df.groupby(['sub_center_id', 'camp_id'])
  bp = grouped.agg(
    Tot_num_patients = pd.NamedAgg(column = 'ANC_Mother Id', aggfunc= lambda x: len(list(x))),
    Num_camps = pd.NamedAgg(column = 'cluster_date', aggfunc= lambda x: len(set(x))),
    dates = pd.NamedAgg(column = 'cluster_date', aggfunc= lambda x: list(set(x)))
  ).reset_index()

  anmgrouped = bp.groupby(['sub_center_id'])
  bpanm = anmgrouped.agg(
    Tot_num_patients = pd.NamedAgg(column = 'Tot_num_patients', aggfunc= sum),
    Num_camps = pd.NamedAgg(column = 'Num_camps', aggfunc= sum),
    Num_locations = pd.NamedAgg(column = 'camp_id', aggfunc= lambda x: len(list(x))),
    dates = pd.NamedAgg(column = 'dates', aggfunc= lambda x: list(x))
  ).reset_index()

  dates_diff = np.array([])
  for i in range(len(bpanm)):    
    num_camps = bpanm.loc[i,'Num_camps']
    if num_camps<2:
      dates_diff = np.append(dates_diff,28)
    else:
      anm_dates = bpanm.loc[i,'dates']
      flat_list = [item for sublist in anm_dates for item in sublist]
      flat_list.sort()
      anm_diff = np.array([])
      for j in range(1,len(flat_list)):
        diff = (flat_list[j] - flat_list[j-1]).days
        anm_diff = np.append(anm_diff,diff)
      dates_diff = np.append(dates_diff, np.mean(anm_diff))

  dates_diff = pd.Series(dates_diff)
  bpanm = bpanm.assign(dates_diff=dates_diff.values)

  # merge with dummy dataframe
  df_merge = pd.merge(dummydf, bpanm, on='sub_center_id', how='outer')
  
  t = df_merge['Tot_num_patients'].isna().sum()
  df_merge['Tot_num_patients'].fillna(0, inplace=True)
  df_merge['Num_camps'].fillna(0, inplace=True)
  df_merge['dates_diff'].fillna(0, inplace=True)
  df_merge['Num_locations'].fillna(0, inplace=True)

  df_merge.to_csv('test2.csv')

  window_features = np.asarray([np.asarray(df_merge['Tot_num_patients']), np.asarray(df_merge['Num_camps']), np.asarray(df_merge['dates_diff']), np.asarray(df_merge['Num_locations'])])

  return np.transpose(window_features), t

# Gets the fraud probabilities for each rule by running a sliding window
def get_fraud_probabilities_new(window_size_weeks = 4):
  
  camp_level_rules = [getBPsdRuleData, getModfUrineRule, getBPhyperRule, getHBrule,
    getHB7rule, bpcontradictionRule, weightcontradictionRule, hbcontradictionRule, 
    bloodsugarcontradictionRule, fetalhrcontradictionRule, urinesugarcontradictionRule, albumincontradictionRule]
  patient_level_rules = [getHIVnotdoneRule, getBloodGnotdoneRule, getVDRLnotdoneRule, getFundalHeightRule, getebfRule]
  patient_dates_columns = ['ANC_EDD','ANC_EDD','ANC_EDD','ANC_EDD', 'Delivery_Delivery Date']

  # old data
  df_old, dftwins_old = read_data()
  # new data
  df, dftwins = read_data_new()
  all_campdate_df = get_campdate_level_df(df)
  sliding_dates = get_sliding_dates_new(window_size_weeks)
  dummydf = get_dummy_anm_df(df_old)
  print(sliding_dates)

  # combining data
  df_new = filter_new_data(df)
  dftwins_new = filter_new_data(dftwins)
  df_all = pd.concat([df_old, df_new])
  dftwins_all = pd.concat([dftwins_old, dftwins_new])

  # windows * rules * anm probabilities
  window_probabilities = []
  features = []
  t1 = []
  for i in range(len(sliding_dates[1])):
  #for i in range(2):
    rule_probabilities = []
    end_date = sliding_dates[1][i]

    # camp date level rules
    start_date = end_date + relativedelta(weeks=-window_size_weeks)
    campdate_df = get_window_df(all_campdate_df, start_date, end_date)
    for rule in camp_level_rules:
      probabilities = rule(campdate_df, dummydf)
      rule_probabilities.append(probabilities)

    # for patient level rules
    for j in range(len(patient_level_rules)):
      p_start_date = end_date + relativedelta(days=-224)
      patient_df = get_window_df(df_all, p_start_date, end_date, column=patient_dates_columns[j])
      probabilities = patient_level_rules[j](patient_df,dummydf)
      rule_probabilities.append(probabilities)

    # for death rule
    d_start_date = end_date + relativedelta(days=-224)
    patient_df = get_window_df(dftwins_all, d_start_date, end_date, column='ANC_Date Of Outcome')
    probabilities = deathRule(patient_df,dummydf)
    rule_probabilities.append(probabilities)

    window_probabilities.append(rule_probabilities)

    window_features, t = get_window_features(campdate_df, dummydf)
    t1.append(t)
    features.append(window_features)
  
  window_probabilities = np.array(window_probabilities)
  features = np.asarray(features)
  print(window_probabilities.shape)
  print(features.shape)
  print(np.sum(t1))

  return window_probabilities, features

# fraud probability vector history for each point
def getHistory_new(test_fraud_prob, test_fraud_prob_new, num_months=6, num_anm=85):
  all_history = np.append(test_fraud_prob, test_fraud_prob_new, axis=0)
  print(all_history.shape)
  ign = len(test_fraud_prob)
  num_points = len(test_fraud_prob_new)
  fraud_vect_history = []
  for i in range(num_points):
    idx = ign + i
    h = []
    for j in range(1,num_months+1):
      k = num_months + 1 - j
      vect = all_history[idx-k*num_anm]
      h.append(vect)
    fraud_vect_history.append(h)
  
  fraud_vect_history = np.asarray(fraud_vect_history)
  print(fraud_vect_history.shape)

  return fraud_vect_history

fraud_probabilities_new, features_new = get_fraud_probabilities_new()
# Reshape
fraud_prob_new = fraud_probabilities_new.swapaxes(1,2)
shape2 = fraud_prob_new.shape
print(shape2)
test_fraud_prob_new = fraud_prob_new.reshape(shape2[0]*shape2[1],shape2[2])
new_shape2 = test_fraud_prob_new.shape
print(new_shape2)
 
fshape = features_new.shape
valid_inp1_new = features_new.reshape(fshape[0]*fshape[1],fshape[2])
print(features_new.shape, valid_inp1_new.shape)

fraud_hist = getHistory_new(test_fraud_prob,test_fraud_prob_new)

maskjan = [False, False, False, True, True, True, False, True, False, True, False, True, True, True, True, False, True, False, True, False, False, True, False, True, True, False, True, True, True, True, False, True, False, False, False, True, True, True, True, False, False, False, True, True, False, True, True, True, False, True, True, False, True, True, False, False, True, True, True, True, True, True, True, True, True, True, False, False, False, True, False, True, True, False, True, True, True, True, True, True, True, True, True, True, True]
maskfeb = [False, False, False, True, True, True, False, True, False, True, True, True, True, True, True, False, True, True, True, False, False, True, False, True, True, False, True, True, False, True, False, True, False, False, False, False, False, False, True, False, False, False, True, False, False, True, True, True, False, False, True, False, True, True, False, False, True, True, False, False, True, False, False, False, True, True, False, False, True, True, False, True, True, False, True, True, True, True, True, False, True, True, True, True, True]
maskmarch = [False, False, False, True, True, True, False, True, False, True, False, True, True, True, False, False, True, False, False, False, False, False, False, True, False, False, True, True, False, True, False, True, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, True, False, True, False, True, False, True, True, False, False, False, False, True, False, True, False, False, False, False, False, True, True, True, True, False, False, False, True]

"""Variational Autoencoder"""

tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions



print(tf.__version__)
print(tfp.__version__)

# global variables
start_time = time.time()
intermediary_dims = [5,3]
latent_dim = 2
batch_size = 8
max_epochs = 1000
reconstruct_samples_n = 100

def vaeModel3(input_shape):
  original_dim = input_shape
  prior = tfd.MultivariateNormalDiag(
    loc=tf.zeros([latent_dim]),
    scale_identity_multiplier=1.0)

  encoder = tfk.Sequential([
    tfkl.InputLayer(input_shape=input_shape, name='encoder_input'),
    tfkl.Dense(tfpl.MultivariateNormalTriL.params_size(latent_dim), activation=tf.nn.leaky_relu),
    tfpl.MultivariateNormalTriL(latent_dim,
                                activity_regularizer=tfpl.KLDivergenceRegularizer(prior)),
  ], name='encoder')

  encoder.summary()

  decoder = tfk.Sequential([
    tfkl.InputLayer(input_shape=[latent_dim]),
    tfkl.Dense(tfpl.IndependentNormal.params_size(original_dim), activation=tf.nn.leaky_relu),
    tfpl.IndependentNormal(original_dim),
  ], name='decoder')

  decoder.summary()

  vae = tfk.Model(inputs=encoder.inputs,
                outputs=decoder(encoder.outputs[0]),
                name='vae_mlp')

  negloglik = lambda x, rv_x: -rv_x.log_prob(x)

  vae.compile(optimizer=tf.keras.optimizers.Nadam(),
            loss=negloglik)

  vae.summary()

  return encoder, decoder, vae

def trainModel(trainX, vae):

  train_sample, val_sample = train_test_split(trainX, test_size=0.2)

  tf_train = tf.data.Dataset.from_tensor_slices((train_sample, train_sample)).batch(batch_size).prefetch(
    tf.data.experimental.AUTOTUNE).shuffle(int(10e4))
  tf_val = tf.data.Dataset.from_tensor_slices((val_sample, val_sample)).batch(batch_size).prefetch(
    tf.data.experimental.AUTOTUNE).shuffle(int(10e4))

  checkpointer = ModelCheckpoint(filepath='bestmodel.h5', verbose=0, save_best_only=True)
  earlystopper = EarlyStopping(monitor='val_loss', mode='min', min_delta=0.005, patience=20, verbose=0,
                             restore_best_weights=True)

  hist = vae.fit(tf_train,
               epochs=max_epochs,
               shuffle=False,
               verbose=2,
               validation_data=tf_val,
               callbacks=[checkpointer, earlystopper])

  plot_loss(hist)

### Utility Functions

# Plot Keras training history
def plot_loss(hist):
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.yscale('log', basey=10)
    plt.savefig("modelloss");
    plt.show()
    plt.clf()


def reconstruction_prob(eval_samples, reconstruct_samples_n, encoder, decoder):
    encoder_out = encoder(eval_samples)
    encoder_samples = encoder_out.sample(reconstruct_samples_n)
    res = np.mean(decoder(encoder_samples).prob(eval_samples), axis=0)
    return res

def reconstruction_log_prob(eval_samples, reconstruct_samples_n, encoder, decoder):
    encoder_out = encoder(eval_samples)
    encoder_samples = encoder_out.sample(reconstruct_samples_n)
    res = np.mean(decoder(encoder_samples).log_prob(eval_samples), axis=0)
    print("log decoder")
    print(decoder(encoder_samples))

    return res

def both_prob(eval_samples, reconstruct_samples_n, encoder, decoder):
    encoder_out = encoder(eval_samples)
    encoder_samples = encoder_out.sample(reconstruct_samples_n)
    res = np.mean(decoder(encoder_samples).prob(eval_samples), axis=0)
    res_log = np.mean(decoder(encoder_samples).log_prob(eval_samples), axis=0)
    return res, res_log

def test_visualize(X, encoder, decoder, p):
    x_prob, x_log_prob = both_prob(X, reconstruct_samples_n, encoder, decoder)

    plt.hist(x_prob, bins=60)
    plt.title("prob")
    plt.show()
    plt.hist(x_log_prob, bins=60)
    plt.title("log")
    plt.show()
    plt.hist(np.log(x_prob), bins=60)
    plt.show()

def dense_layers(sizes):
    return tfk.Sequential([tfkl.Dense(size, activation=tf.nn.leaky_relu) for size in sizes])

# graphs
def visualize(X, encoder, decoder, p):
    
    x_prob = reconstruction_prob(X, reconstruct_samples_n, encoder, decoder)
    # ax = plt.hist([x_prob[Y[:, 0] == 0], x_prob[Y[:, 0] == 1]], bins=60, label=['Good', 'Bad'])
    ax = plt.hist(x_prob, bins=60)
    # plt.legend(loc='upper right')
    plt.title(p + ' reconstruction probability')
    plt.ylabel('frequency')
    plt.xlabel("p(x|x')")
    plt.savefig(p + "reconstruction.png")
    plt.show()
    plt.clf()

    print(x_prob)

    x_log_prob = reconstruction_log_prob(X, reconstruct_samples_n, encoder, decoder)
    # ax = plt.hist([x_prob[Y[:, 0] == 0], x_prob[Y[:, 0] == 1]], bins=60, label=['Good', 'Bad'])
    ax = plt.hist(x_log_prob, bins=60)
    # plt.legend(loc='upper right')
    plt.title('Reconstruction log probability in ' + p)
    plt.ylabel('frequency')
    plt.xlabel("log p(x|x')")
    plt.savefig(p + "reconstruction_log.png")
    plt.show()
    plt.clf()

    print(x_log_prob)


#train - all rules
trainX = test_fraud_prob[0:85*25]
input_shape = trainX.shape[1]
encoder, decoder, vae = vaeModel3(input_shape)
trainModel(trainX, vae)
print(trainX.shape)

# test - all rules
testX = test_fraud_prob[85*32:85*33]
visualize(testX[maskjan], encoder,decoder, "January")
testX = test_fraud_prob[85*33:85*34]
visualize(testX[maskfeb], encoder,decoder, "February")
testX = test_fraud_prob[85*34:85*35]
visualize(testX[maskmarch], encoder,decoder, "March")

print("end")

# recon. graphs for train data
visualize(trainX, encoder, decoder, 'train')

# graphs from same encoder samples of test data
testX = test_fraud_prob[85*32:85*33]
test_visualize(testX[maskjan], encoder,decoder, "Jan")
testX = test_fraud_prob[85*33:85*34]
test_visualize(testX[maskfeb], encoder,decoder, "Feb")
testX = test_fraud_prob[85*34:85*35]
test_visualize(testX[maskmarch], encoder,decoder, "March")

print("end")

# for all field test months
visualize(test_fraud_prob_new, encoder, decoder, 'train')

# vae for bp rule only
trainX = test_fraud_prob[0:85*25,0:1]
input_shape = trainX.shape[1]
encoder, decoder, vae = vaeModel3(input_shape)
trainModel(trainX, vae)
print(trainX.shape)

testX = test_fraud_prob[85*32:85*33,0:1]
visualize(testX[maskjan], encoder,decoder, "Jan_bp")
testX = test_fraud_prob[85*33:85*34,0:1]
visualize(testX[maskfeb], encoder,decoder, "Feb_bp")
testX = test_fraud_prob[85*34:85*35,0:1]
visualize(testX[maskmarch], encoder,decoder, "March_bp")

print("end")



