import pandas as pd
import numpy as np

from .kdes import KDEs


class LongRangeRules:
    """
      This class processes the long range rules and outputs the non-diligence probability vectors
      of ANMs in a time frame using KDE.

      Long range rules are the rules that track longer term phenomenon,
      and hence calculated over a period of few months (6).

      At initialization, the KDEs are obtained using R scripts. Percentages are obtained after processing each rule by
      running a sliding window and they are converted to non-diligence probability using KDE, by taking the area under
      the curve from 0 to x or x to 100 according to the nature of the rule.

      """

    def __init__(self, kdes):

        self.func_get_prob_mass_trans = kdes.func_get_prob_mass_trans
        self.death_fit = kdes.death_fit
        self.ebf_fit = kdes.ebf_fit
        self.hiv_fit = kdes.hiv_fit
        self.bloodg_fit = kdes.bloodg_fit
        self.vdrl_fit = kdes.vdrl_fit
        self.fundalh_fit = kdes.fundalh_fit





    def deathRule(self, dfr, dummydf):
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
        dfmod = dfr.loc[dfr['ANC_Mother Id'].notnull(), ['sub_center_id', 'camp_id', 'ANC_Mother Id',
                                                         'Delivery_Pregnancy Out Come']]

        # Step 2
        grouped = dfmod.groupby(['sub_center_id'])
        dr = grouped.agg(
            live_birth_count=pd.NamedAgg(column='Delivery_Pregnancy Out Come',
                                         aggfunc=lambda x: tuple(x.dropna()).count(1)),
            still_birth_count=pd.NamedAgg(column='Delivery_Pregnancy Out Come',
                                          aggfunc=lambda x: tuple(x.dropna()).count(2)),
            abortion_count=pd.NamedAgg(column='Delivery_Pregnancy Out Come',
                                       aggfunc=lambda x: tuple(x.dropna()).count(3))
        ).reset_index()

        dr['death_percentage'] = dr.apply(
            lambda row: row['still_birth_count'] / (row['live_birth_count'] + row['still_birth_count']) * 100 if (row[
                                                                                                                      'live_birth_count'] +
                                                                                                                  row[
                                                                                                                      'still_birth_count']) != 0 else 0,
            axis=1)

        # Step 3
        fraud_probabilities = []
        for i in range(len(dr)):
            x = float(dr.loc[i, 'death_percentage'])
            prob = np.asarray(self.func_get_prob_mass_trans(self.death_fit, x, 100))
            fraud_probabilities.append(prob[0])

        fraud_probabilities = pd.Series(fraud_probabilities)
        dr = dr.assign(fraud_probabilities=fraud_probabilities.values)

        # merge with dummy dataframe
        df_merge = pd.merge(dummydf, dr, on='sub_center_id', how='left')
        df_merge['fraud_probabilities'].fillna(df_merge['fraud_probabilities'].mean(), inplace=True)

        return list(df_merge['fraud_probabilities'])

    def getHIVnotdoneRule(self, df, dummydf):
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
        dfmod = df.loc[
            df['ANC_Mother Id'].notnull(), ['sub_center_id', 'camp_id', 'ANC_Mother Id', 'ANC_Hiv Test Result']]

        # Step 2
        bl = dfmod.groupby(['sub_center_id']).agg({'ANC_Mother Id': lambda x: len(x),
                                                   'ANC_Hiv Test Result': lambda x: len(x) - len(
                                                       x.dropna())}).reset_index()
        bl['blankpercentage'] = bl.apply(lambda row: row['ANC_Hiv Test Result'] / row['ANC_Mother Id'] * 100, axis=1)

        # Step 3
        fraud_probabilities = []
        for i in range(len(bl)):
            x = float(bl.loc[i, 'blankpercentage'])
            prob = np.asarray(self.func_get_prob_mass_trans(self.hiv_fit, 0, x))
            fraud_probabilities.append(prob[0])

        fraud_probabilities = pd.Series(fraud_probabilities)
        bl = bl.assign(fraud_probabilities=fraud_probabilities)

        # merge with dummy dataframe
        df_merge = pd.merge(dummydf, bl, on='sub_center_id', how='left')
        df_merge['fraud_probabilities'].fillna(np.mean(df_merge['fraud_probabilities']), inplace=True)

        return list(df_merge['fraud_probabilities'])

    def getBloodGnotdoneRule(self, df, dummydf):
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
        bl['blankpercentage'] = bl.apply(lambda row: row['ANC_Blood Group'] / row['ANC_Mother Id'] * 100, axis=1)

        # Step 3
        fraud_probabilities = []
        for i in range(len(bl)):
            x = float(bl.loc[i, 'blankpercentage'])
            prob = np.asarray(self.func_get_prob_mass_trans(self.bloodg_fit, 0, x))
            fraud_probabilities.append(prob[0])

        fraud_probabilities = pd.Series(fraud_probabilities)
        bl = bl.assign(fraud_probabilities=fraud_probabilities.values)

        # merge with dummy dataframe
        df_merge = pd.merge(dummydf, bl, on='sub_center_id', how='left')
        df_merge['fraud_probabilities'].fillna(df_merge['fraud_probabilities'].mean(), inplace=True)

        return list(df_merge['fraud_probabilities'])

    def getVDRLnotdoneRule(self, df, dummydf):
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
        dfmod = df.loc[
            df['ANC_Mother Id'].notnull(), ['sub_center_id', 'camp_id', 'ANC_Mother Id', 'ANC_Vdrl Test Result']]

        # Step 2
        bl = dfmod.groupby(['sub_center_id']).agg({'ANC_Mother Id': lambda x: len(x),
                                                   'ANC_Vdrl Test Result': lambda x: len(x) - len(
                                                       x.dropna())}).reset_index()
        bl['blankpercentage'] = bl.apply(lambda row: row['ANC_Vdrl Test Result'] / row['ANC_Mother Id'] * 100, axis=1)

        # Step 3
        fraud_probabilities = []
        for i in range(len(bl)):
            x = float(bl.loc[i, 'blankpercentage'])
            prob = np.asarray(self.func_get_prob_mass_trans(self.vdrl_fit, 0, x))
            fraud_probabilities.append(prob[0])

        fraud_probabilities = pd.Series(fraud_probabilities)
        bl = bl.assign(fraud_probabilities=fraud_probabilities.values)

        # merge with dummy dataframe
        df_merge = pd.merge(dummydf, bl, on='sub_center_id', how='left')
        df_merge['fraud_probabilities'].fillna(df_merge['fraud_probabilities'].mean(), inplace=True)

        return list(df_merge['fraud_probabilities'])

    def getFundalHeightRule(self, df, dummydf):
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
        dfmod = df.loc[df['ANC_Mother Id'].notnull(), ['sub_center_id', 'camp_id', 'ANC_Mother Id',
                                                       'ANC_ANC1 Fundal Height', 'ANC_ANC2 Fundal Height',
                                                       'ANC_ANC3 Fundal Height', 'ANC_ANC4 Fundal Height']]

        # Step 2
        dfmod['fraud1'] = dfmod.apply(lambda row: 1 if (
                row['ANC_ANC1 Fundal Height'] == 2 and row['ANC_ANC2 Fundal Height'] == 2 and row[
            'ANC_ANC3 Fundal Height'] == 2) else 0, axis=1)
        dfmod['fraud2'] = dfmod.apply(lambda row: 1 if (
                row['ANC_ANC2 Fundal Height'] == 2 and row['ANC_ANC3 Fundal Height'] == 2 and row[
            'ANC_ANC4 Fundal Height'] == 2) else 0, axis=1)
        dfmod['fraudFundalH'] = dfmod.apply(lambda row: 1 if (row['fraud1'] == 1 or row['fraud2'] == 1) else 0, axis=1)
        dfanm3 = dfmod.groupby(['sub_center_id']).agg({'ANC_Mother Id': lambda x: len(x),
                                                       'fraudFundalH': lambda x: tuple(x)}).reset_index()
        dfanm3['suspicious_percentage'] = dfanm3.apply(
            lambda row: row['fraudFundalH'].count(1) / len(row['fraudFundalH']) * 100 if len(
                row['fraudFundalH']) != 0 else 0, axis=1)

        # Step 3
        fraud_probabilities = []
        for i in range(len(dfanm3)):
            x = float(dfanm3.loc[i, 'suspicious_percentage'])
            prob = np.asarray(self.func_get_prob_mass_trans(self.fundalh_fit, 0, x))
            fraud_probabilities.append(prob[0])

        fraud_probabilities = pd.Series(fraud_probabilities)
        dfanm3 = dfanm3.assign(fraud_probabilities=fraud_probabilities.values)

        # merge with dummy dataframe
        df_merge = pd.merge(dummydf, dfanm3, on='sub_center_id', how='left')
        df_merge['fraud_probabilities'].fillna(df_merge['fraud_probabilities'].mean(), inplace=True)

        return list(df_merge['fraud_probabilities'])

    def getebfRule(self, df, dummydf):
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
        dfmod = df.loc[
            df['Delivery_Delivery Outcome ID'].notnull(), ['sub_center_id', 'camp_id', 'Delivery_Delivery Outcome ID',
                                                           'Delivery_Breastfeeding from birth']]

        # Step 2
        grouped = dfmod.groupby(['sub_center_id'])
        ebf = grouped.agg(
            val=pd.NamedAgg(column='Delivery_Breastfeeding from birth', aggfunc=lambda x: tuple(x.dropna())),
            val_count=pd.NamedAgg(column='Delivery_Breastfeeding from birth', aggfunc=lambda x: len(tuple(x.dropna()))),
            true_count=pd.NamedAgg(column='Delivery_Breastfeeding from birth',
                                   aggfunc=lambda x: tuple(x.dropna()).count(True))
        ).reset_index()
        ebf['true_percentage'] = ebf.apply(
            lambda row: row['true_count'] / row['val_count'] * 100 if row['val_count'] != 0 else 0, axis=1)

        # Step 3
        fraud_probabilities = []
        for i in range(len(ebf)):
            x = float(ebf.loc[i, 'true_percentage'])
            prob = np.asarray(self.func_get_prob_mass_trans(self.ebf_fit, 0, x))
            fraud_probabilities.append(prob[0])

        fraud_probabilities = pd.Series(fraud_probabilities)
        ebf = ebf.assign(fraud_probabilities=fraud_probabilities.values)

        # merge with dummy dataframe
        df_merge = pd.merge(dummydf, ebf, on='sub_center_id', how='left')
        df_merge['fraud_probabilities'].fillna(df_merge['fraud_probabilities'].mean(), inplace=True)

        return list(df_merge['fraud_probabilities'])
