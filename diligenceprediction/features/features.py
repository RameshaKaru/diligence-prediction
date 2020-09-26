from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np

from ..ruleskde.longrangerules import LongRangeRules
from ..ruleskde.shortrangerules import ShortRangeRules


class Features:
    """
    This class handles obtaining the features; i.e the non diligence probability vectors of ANMs and meta data of ANMs

    """

    def __init__(self, allANMdf, kdes, sliding_dates, all_campdate_df, df_all, dftwins_all, add_lookback_days):
        """
        Initializes the features class with parameters related to preprocessed data

        Parameters
        ----------
        allANMdf : dataframe
            dataframe with all sub center ids
        kdes : KDE object
            KDEs of 18 rules
        sliding_dates :  list
            sliding window dates (only the end date of each slide is in the list)
        all_campdate_df, df_all, dftwins_all: dataframe
            preprocessed dataframes
        add_lookback_days: list
            list of additional days to look back for long range rules (if ignore date ranges are specified)

        """
        self.allANMdf = allANMdf
        self.kdes = kdes
        self.sliding_dates = sliding_dates
        self.all_campdate_df = all_campdate_df
        self.df_all = df_all
        self.dftwins_all = dftwins_all
        self.add_lookback_days = add_lookback_days


    def get_fraud_probabilities(self, window_size_weeks=4):
        """
        Calculates the non-diligence probabilities for each rule for each ANM by running a sliding window

        Parameters
        ----------
        window_size_weeks : int
            sliding window size

        Returns
        -------
        window_probabilities : non-diligence probabilities for each window each rule each ANM
        features: meta features of each ANM in each sliding window

        """

        longRules = LongRangeRules(self.kdes)
        shortRules = ShortRangeRules(self.kdes)

        camp_level_rules = [shortRules.getBPsdRuleData, shortRules.getModfUrineRule, shortRules.getBPhyperRule,
                            shortRules.getHBrule,
                            shortRules.getHB7rule, shortRules.bpcontradictionRule, shortRules.weightcontradictionRule,
                            shortRules.hbcontradictionRule,
                            shortRules.bloodsugarcontradictionRule, shortRules.fetalhrcontradictionRule,
                            shortRules.urinesugarcontradictionRule,
                            shortRules.albumincontradictionRule]
        patient_level_rules = [longRules.getHIVnotdoneRule, longRules.getBloodGnotdoneRule,
                               longRules.getVDRLnotdoneRule,
                               longRules.getFundalHeightRule, longRules.getebfRule]
        patient_dates_columns = ['ANC_EDD', 'ANC_EDD', 'ANC_EDD', 'ANC_EDD', 'Delivery_Delivery Date']

        # windows * rules * anm probabilities
        window_probabilities = []
        features = []
        t1 = []
        for i in range(len(self.sliding_dates)):
            rule_probabilities = []
            end_date = self.sliding_dates[i]
            lookback = self.add_lookback_days[i]

            # camp date level rules
            start_date = end_date + relativedelta(weeks=-window_size_weeks)
            campdate_df = self.get_window_df(self.all_campdate_df, start_date, end_date)
            for rule in camp_level_rules:
                probabilities = rule(campdate_df, self.allANMdf)
                rule_probabilities.append(probabilities)

            # for patient level rules
            for j in range(len(patient_level_rules)):
                p_start_date = end_date + relativedelta(months=-6) + relativedelta(days=-lookback)
                patient_df = self.get_window_df(self.df_all, p_start_date, end_date, column=patient_dates_columns[j])
                probabilities = patient_level_rules[j](patient_df, self.allANMdf)
                rule_probabilities.append(probabilities)

            # for death rule
            d_start_date = end_date + relativedelta(months=-6) + relativedelta(days=-lookback)
            patient_df = self.get_window_df(self.dftwins_all, d_start_date, end_date, column='ANC_Date Of Outcome')
            probabilities = longRules.deathRule(patient_df, self.allANMdf)
            rule_probabilities.append(probabilities)

            window_probabilities.append(rule_probabilities)

            window_features, t = self.get_window_features(campdate_df)
            t1.append(t)
            features.append(window_features)

        window_probabilities = np.array(window_probabilities)
        features = np.asarray(features)
        print(window_probabilities.shape)

        return window_probabilities, features

    def get_window_features(self, campdate_df):
        """
        Finds meta data of each ANM in each time window

        Parameters
        ----------
        campdate_df : dataframe
            preprocessed dataframe

        """

        grouped = campdate_df.groupby(['sub_center_id', 'camp_id'])
        bp = grouped.agg(
            Tot_num_patients=pd.NamedAgg(column='ANC_Mother Id', aggfunc=lambda x: len(list(x))),
            Num_camps=pd.NamedAgg(column='cluster_date', aggfunc=lambda x: len(set(x))),
            dates=pd.NamedAgg(column='cluster_date', aggfunc=lambda x: list(set(x)))
        ).reset_index()

        anmgrouped = bp.groupby(['sub_center_id'])
        bpanm = anmgrouped.agg(
            Tot_num_patients=pd.NamedAgg(column='Tot_num_patients', aggfunc=sum),
            Num_camps=pd.NamedAgg(column='Num_camps', aggfunc=sum),
            Num_locations=pd.NamedAgg(column='camp_id', aggfunc=lambda x: len(list(x))),
            dates=pd.NamedAgg(column='dates', aggfunc=lambda x: list(x))
        ).reset_index()

        dates_diff = np.array([])
        for i in range(len(bpanm)):
            num_camps = bpanm.loc[i, 'Num_camps']
            if num_camps < 2:
                dates_diff = np.append(dates_diff, 28)
            else:
                anm_dates = bpanm.loc[i, 'dates']
                flat_list = [item for sublist in anm_dates for item in sublist]
                flat_list.sort()
                anm_diff = np.array([])
                for j in range(1, len(flat_list)):
                    diff = (flat_list[j] - flat_list[j - 1]).days
                    anm_diff = np.append(anm_diff, diff)
                dates_diff = np.append(dates_diff, np.mean(anm_diff))

        dates_diff = pd.Series(dates_diff)
        bpanm = bpanm.assign(dates_diff=dates_diff.values)

        # merge with dummy dataframe
        df_merge = pd.merge(self.allANMdf, bpanm, on='sub_center_id', how='left')

        t = df_merge['Tot_num_patients'].isna().sum()
        df_merge['Tot_num_patients'].fillna(0, inplace=True)
        df_merge['Num_camps'].fillna(0, inplace=True)
        df_merge['dates_diff'].fillna(0, inplace=True)
        df_merge['Num_locations'].fillna(0, inplace=True)

        window_features = np.asarray([np.asarray(df_merge['Tot_num_patients']), np.asarray(df_merge['Num_camps']),
                                      np.asarray(df_merge['dates_diff']), np.asarray(df_merge['Num_locations'])])

        return np.transpose(window_features), t

    def get_window_df(self, df, start_date, end_date, column='cluster_date'):
        """
        Extracts a dataframe within a given date range

        Parameters
        ----------
        df : dataframe
            full dataframe to be processed
        start_date : date
            start date
        end_date : date
            end date
        column: string
            column based on which the dates should be extracted

        Returns
        -------
        dataframe within the given time range

        """
        df[column] = pd.to_datetime(df[column])
        mask = (df[column] > pd.to_datetime(start_date)) & (df[column] <= pd.to_datetime(end_date))
        new_df = df.loc[mask]

        return new_df