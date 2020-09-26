from .processhelpers.script_helpers import get_saved_model, get_cluster_centers, read_data, get_configs, get_all_ANM
from .processhelpers.script_helpers import get_predict_ANM, get_campdate_level_df, filter_new_data, get_ignore_dates, \
    get_overlap, get_no_ignore_sliding_dates
from .ruleskde.kdes import KDEs
from .features.features import Features

import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np
from skfuzzy.cluster import cmeans, cmeans_predict


class HDPS:
    """
        This is the main class of the diligence prediction system.
        This class preprocess the data to convert them to a required format for the rules
        Obtains the non diligence probability vectors of last windows
        Predicts the non diligence scores for the next time window using the saved model
        Also allows obtaining the actual non diligence scores using the saved cmeans cluster centers

    """

    def __init__(self):
        """
        During initialization:
        - gets the configurations from the config file
        - gets the KDEs
        - gets the saved model
        - gets the other required data

        """
        print("INITIALIZING")
        self.configs = get_configs()
        self.cmeans_cntr = get_cluster_centers()
        self.kdes = KDEs()
        self.model = get_saved_model()
        self.allANMdf = get_all_ANM()
        self.num_files = len(self.configs['datafiles'])
        self.predictANMids = get_predict_ANM(self.configs["next_sub_center_ids_file"])
        self.ignore_dates = get_ignore_dates(self.configs['ignore_dates'])

    def preprocess_data(self):
        """
        This function preprocesses the data and obtain non diligence vectors of the ANMs in each time window
        """

        print("PREPROCESSING DATA")

        # sliding windows
        sliding_dates_list_new = self.get_sliding_dates_new()
        print(sliding_dates_list_new)
        add_lookback_days_new = self.get_add_lookback_days_new(sliding_dates_list_new)
        print(add_lookback_days_new)

        # read all original dataframes
        df_list, dftwins_list = self.get_all_df()

        # dataframes for short range rules
        all_campdate_df_list = self.get_campdate_df_list(df_list)
        all_campdate_df = self.get_filtered_campdate_df(all_campdate_df_list)

        # dataframes for long range rules
        df_all_list = self.prepare_long_df(df_list)
        dftwins_all_list = self.prepare_long_df(dftwins_list)

        print("GETTING FEATURES")

        features_obj = Features(allANMdf=self.allANMdf,
                                kdes=self.kdes,
                                sliding_dates=sliding_dates_list_new,
                                all_campdate_df=all_campdate_df,
                                df_all=df_all_list[-1],
                                dftwins_all=dftwins_all_list[-1],
                                add_lookback_days=add_lookback_days_new)
        fraud_probabilities, meta_features = features_obj.get_fraud_probabilities()

        fraud_prob = fraud_probabilities.swapaxes(1, 2)
        shape2 = fraud_prob.shape
        test_fraud_prob = fraud_prob.reshape(shape2[0] * shape2[1], shape2[2])

        history_vect = self.get_history(test_fraud_prob)

        return test_fraud_prob, history_vect, meta_features

    def get_past_scores(self, test_fraud_prob, meta_features, num_anm=85):
        """
        This function gets the actual scores of the last 6 months using the saved cmeans cluster centers
        and saves them to csvs in outputs folder
        """

        print("GETTING ACTUAL PAST SCORES")
        u, _, _, _, _, _ = cmeans_predict(test_fraud_prob.T, self.cmeans_cntr, m=2, error=0.005, maxiter=1000)
        labels = u.T

        for i in range(len(meta_features)):
            window_num_patients = meta_features[i, :, 0]
            window_scores = labels[i * num_anm: (i + 1) * num_anm, 1]
            mask = []
            for j in range(num_anm):
                if window_num_patients[j] > 0:
                    mask.append(True)
                else:
                    mask.append(False)
            anm_mask = np.array(self.allANMdf['sub_center_id'])[mask]
            window_scores_mask = window_scores[mask]

            scores_df = pd.DataFrame({'sub_center_id': anm_mask, 'scores': window_scores_mask})
            scores_df.to_csv('outputs/past_scores' + str(i) + '.csv')

    def get_history(self, fraud_prob_vectors, num_months=6, num_anm=85):
        """
        Gets the non diligence probability vector history for each point

        Parameters
        ----------
        fraud_prob_vectors : array
            non diligence probability vectors
        num_months : int
            number of months in history to be considered
        num_anm : int
            number of ANMs

        Returns
        -------
        non diligence probability vector arrays of ANMs for each ANM each predictable window

        """
        ign = num_anm * num_months
        num_points = len(fraud_prob_vectors) - ign + num_anm
        fraud_vect_history = []
        for i in range(num_points):
            idx = ign + i
            h = []
            for j in range(1, num_months + 1):
                k = num_months + 1 - j
                vect = fraud_prob_vectors[idx - k * num_anm]
                h.append(vect)
            fraud_vect_history.append(h)

        fraud_vect_history = np.asarray(fraud_vect_history)
        print(fraud_vect_history.shape)

        return fraud_vect_history

    def prepare_long_df(self, df_list):
        """
        Prepares the dataframes for the long range rules, joins the data sets together

        Parameters
        ----------
        df_list : list
            list of dataframes

        Returns
        -------
        list of prepared dataframes

        """
        new_df_list = [df_list[0]]
        df_old = df_list[0]

        for i in range(1, self.num_files):
            online_day = datetime.datetime.strptime(self.configs['datafiles'][i]['start_date'], '%Y, %m, %d')
            df_new = filter_new_data(df_list[i], online_day)
            df_old = pd.concat([df_old, df_new])
            new_df_list.append(df_old)

        return new_df_list

    def get_campdate_df_list(self, df_list):
        """
        Prepares the dataframes for the short range rules

        Parameters
        ----------
        df_list : list
            list of dataframes

        Returns
        -------
        list of prepared dataframes

        """
        all_campdate_df_list = []
        for i in range(self.num_files):
            online_day = datetime.datetime.strptime(self.configs['datafiles'][i]['start_date'], '%Y, %m, %d')
            all_campdate_df = get_campdate_level_df(df_list[i], online_day)
            all_campdate_df_list.append(all_campdate_df)

        return all_campdate_df_list

    def get_filtered_campdate_df(self, df_list):
        """
        Joins all the preprocessed dataframes and filters them removing date ranges which are to be ignored

        Parameters
        ----------
        df_list : list
            list of dataframes

        Returns
        -------
        filtered dataframe

        """
        df_concat = df_list[0]
        for i in range(1, len(df_list)):
            df_concat = pd.concat([df_concat, df_list[i]])

        df_concat['date'] = pd.to_datetime(df_concat['date'])
        for i in range(len(self.ignore_dates)):
            filt = (df_concat['date'].dt.date >= self.ignore_dates[i]['start_date']) & (
                    df_concat['date'].dt.date <= self.ignore_dates[i]['start_date'])
            df_concat = df_concat[~filt]

        return df_concat

    def get_all_df(self):
        """
        Reads all data sets

        Returns
        -------
        list of dataframes with unique mothers
        list of original dataframes

        """
        df_list = []
        dftwins_list = []
        for i in range(self.num_files):
            fileName = self.configs['datafiles'][i]['location'] + self.configs['datafiles'][i]['name']
            df, dftwins = read_data(fileName)
            df_list.append(df)
            dftwins_list.append(dftwins)

        return df_list, dftwins_list

    def get_sliding_dates_new(self):
        """
        Calculates the sliding windows

        Returns
        -------
        list of dates (end date of each sliding window)

        """
        print("Sliding windows")
        sliding_dates_list = []
        end_day = datetime.datetime.strptime(self.configs['datafiles'][-1]['end_date'], '%Y, %m, %d').date()
        online_day = datetime.datetime.strptime(self.configs['datafiles'][0]['start_date'], '%Y, %m, %d').date()
        start_end_day = online_day + relativedelta(months=+6)

        if self.ignore_dates is None:
            sliding_dates_list = get_no_ignore_sliding_dates(end_day, start_end_day)

        else:
            overlap = 0
            for i in range(len(self.ignore_dates)):
                diff = int((self.ignore_dates[i]['end_date'] - self.ignore_dates[i]['start_date']).days)
                temp_overlap = get_overlap(self.ignore_dates[i]['start_date'], self.ignore_dates[i]['end_date'],
                                           online_day, start_end_day)
                if temp_overlap == 0:
                    continue
                elif temp_overlap == diff:
                    overlap = overlap + temp_overlap
                elif temp_overlap < diff:
                    overlap = overlap + diff
            start_end_day = start_end_day + relativedelta(days=+overlap)

            while len(sliding_dates_list) < 6:
                if end_day < start_end_day:
                    print("Not enough data frames. Please add data of more than 1 year")
                    break

                for i in range(len(self.ignore_dates)):
                    win_start = end_day + relativedelta(weeks=-4)
                    temp_overlap = get_overlap(self.ignore_dates[i]['start_date'], self.ignore_dates[i]['end_date'],
                                               win_start, end_day)
                    if temp_overlap == 0:
                        continue
                    else:
                        end_day = self.ignore_dates[i]['start_date'] + relativedelta(days=-1)

                sliding_dates_list.append(end_day)
                end_day = end_day + relativedelta(weeks=-4)

            sliding_dates_list.reverse()

        return sliding_dates_list

    def get_add_lookback_days_new(self, sliding_dates_list):
        """
        Calculates the number of additional days the window has to look back when processing long range rules
        in case ignore_dates ranges are specified in configs

        Parameters
        ----------
        sliding_dates_list : list
            list of sliding window dates

        Returns
        -------
        list of dates to look back in each sliding window

        """
        get_add_lookback_days = []

        if self.ignore_dates is None:
            temp = np.zeros(len(sliding_dates_list))
            get_add_lookback_days.append(temp)

        else:
            for j in range(len(sliding_dates_list)):
                slide_end_date = sliding_dates_list[j]
                slide_start_date = slide_end_date + relativedelta(months=-6)
                overlap = 0
                for k in range(len(self.ignore_dates)):
                    temp_overlap = get_overlap(self.ignore_dates[k]['start_date'], self.ignore_dates[k]['end_date'],
                                               slide_start_date, slide_end_date)
                    diff = int((self.ignore_dates[k]['end_date'] - self.ignore_dates[k]['start_date']).days)
                    if temp_overlap == 0:
                        continue
                    elif temp_overlap == diff:
                        overlap = overlap + temp_overlap
                    elif temp_overlap < diff:
                        overlap = overlap + diff
                get_add_lookback_days.append(overlap)

        return get_add_lookback_days

    def predict_scores_next(self, history_vect, num_anm=85):
        """
        Predicts the non diligence scores for the next time window and saves them in a csv file

        Parameters
        ----------
        history_vect : array
            history of non diligence probabilities
        num_anm : int
            number of ANMs

        """
        print("PREDICTING")
        last_history_vect = history_vect[-num_anm:, :, :]
        print(last_history_vect.shape)

        y_pred = self.model.predict(last_history_vect)

        if self.predictANMids is None:
            print("IDs of the sub centers where camps will be held is not provided. Hence predicting for all")
            scores_df = pd.DataFrame(
                {'sub_center_id': np.array(self.allANMdf['sub_center_id']), 'scores': y_pred[:, 1]})
            scores_df.to_csv('outputs/scores.csv')
        else:
            next_anm_df = pd.DataFrame(
                {'sub_center_id': self.predictANMids, 'flag': np.full((len(self.predictANMids)), True)})
            df_merge = pd.merge(self.allANMdf, next_anm_df, how="left", on="sub_center_id")
            df_merge['flag'].fillna(False, inplace=True)
            mask = df_merge['flag']
            y_pred_mask = y_pred[mask]
            anm_mask = np.array(self.allANMdf['sub_center_id'])[mask]
            scores_df = pd.DataFrame(
                {'sub_center_id': anm_mask, 'scores': y_pred_mask[:, 1]})
            scores_df.to_csv('outputs/scores.csv')
