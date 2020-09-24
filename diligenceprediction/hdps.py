from .processhelpers.script_helpers import get_saved_model, get_cluster_centers, read_data, get_configs, get_sliding_dates, get_all_ANM
from .processhelpers.script_helpers import get_predict_ANM, get_campdate_level_df, filter_new_data
from .ruleskde.kdes import KDEs
from .features.features import Features

import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np


class HDPS:

    def __init__(self):
        print("INITIALIZING")
        self.configs = get_configs()
        self.cmeans_cntr = get_cluster_centers()
        self.kdes = KDEs()
        self.model = get_saved_model()
        self.allANMdf = get_all_ANM()
        self.num_files = len(self.configs['datafiles'])
        self.predictANMids = get_predict_ANM(self.configs["next_sub_center_ids_file"])

    def preprocess_data(self):
        print("PREPROCESSING DATA")

        # sliding windows
        sliding_dates_list = self.get_sliding_dates_list()

        # read all original dataframes
        df_list, dftwins_list = self.get_all_df()

        # dataframes for short range rules
        all_campdate_df_list = self.get_campdate_df_list(df_list)

        # dataframes for long range rules
        df_all_list = self.prepare_long_df(df_list)
        dftwins_all_list = self.prepare_long_df(dftwins_list)

        print("GETTING FEATURES")

        for i in range(self.num_files):
            features_obj = Features(allANMdf=self.allANMdf,
                                    kdes=self.kdes,
                                    sliding_dates=sliding_dates_list[i],
                                    all_campdate_df=all_campdate_df_list[i],
                                    df_all=df_all_list[i],
                                    dftwins_all=dftwins_all_list[i])
            fraud_probabilities, meta_features = features_obj.get_fraud_probabilities()

            fraud_prob = fraud_probabilities.swapaxes(1, 2)
            shape2 = fraud_prob.shape
            test_fraud_prob_new = fraud_prob.reshape(shape2[0] * shape2[1], shape2[2])

            if i == 0:
                test_fraud_prob = test_fraud_prob_new
            else:
                test_fraud_prob = np.append(test_fraud_prob, test_fraud_prob_new, axis=0)

        # print("Total time-windows processed: ", test_fraud_prob.shape[0]/85)

        history_vect = self.get_history(test_fraud_prob)

        return test_fraud_prob, history_vect

    # fraud probability vector history for each point
    def get_history(self, fraud_prob_vectors, num_months=6, num_anm=85):
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
        new_df_list = [df_list[0]]
        df_old = df_list[0]

        for i in range(1, self.num_files):
            online_day = datetime.datetime.strptime(self.configs['datafiles'][i]['start_date'], '%Y, %m, %d')
            df_new = filter_new_data(df_list[i], online_day)
            df_old = pd.concat([df_old, df_new])
            new_df_list.append(df_old)

        return new_df_list

    def get_campdate_df_list(self, df_list):
        all_campdate_df_list = []
        for i in range(self.num_files):
            online_day = datetime.datetime.strptime(self.configs['datafiles'][i]['start_date'], '%Y, %m, %d')
            all_campdate_df = get_campdate_level_df(df_list[i], online_day)
            all_campdate_df_list.append(all_campdate_df)

        return all_campdate_df_list

    def get_all_df(self):
        df_list = []
        dftwins_list = []
        for i in range(self.num_files):
            fileName = self.configs['datafiles'][i]['location'] + self.configs['datafiles'][i]['name']
            df, dftwins = read_data(fileName)
            df_list.append(df)
            dftwins_list.append(dftwins)

        return df_list, dftwins_list

    def get_sliding_dates_list(self):
        print("Sliding windows")
        sliding_dates_list = []
        for i in range(self.num_files):
            online_day = datetime.datetime.strptime(self.configs['datafiles'][i]['start_date'], '%Y, %m, %d').date()
            data_end_day = datetime.datetime.strptime(self.configs['datafiles'][i]['end_date'], '%Y, %m, %d').date()

            if i == 0:
                online_day = online_day + relativedelta(months=+6) + relativedelta(weeks=-4)

            sliding_dates = get_sliding_dates(online_day=online_day,
                                              data_end_day=data_end_day)
            sliding_dates_list.append(sliding_dates)
            print(sliding_dates)

        return sliding_dates_list

    def predict_scores_next(self, history_vect, num_anm=85):
        print("PREDICTING")
        last_history_vect = history_vect[-num_anm:, :, :]
        print(last_history_vect.shape)

        y_pred = self.model.predict(last_history_vect)
        print(y_pred.shape)

        if self.predictANMids is None:
            print("IDs of the sub centers where camps will be held is not provided. Hence predicting for all")
            scores_df = pd.DataFrame({'sub_center_id': np.array(self.allANMdf['sub_center_id']), 'scores': y_pred[:,1]})
            scores_df.to_csv('outputs/scores.csv')
        else:
            next_anm_df = pd.DataFrame({'sub_center_id': self.predictANMids, 'flag': np.full((len(self.predictANMids)), True)})
            df_merge = pd.merge(self.allANMdf, next_anm_df, how="left", on="sub_center_id")
            df_merge['flag'].fillna(False, inplace=True)
            mask = df_merge['flag']
            y_pred_mask = y_pred[mask]
            anm_mask = np.array(self.allANMdf['sub_center_id'])[mask]
            scores_df = pd.DataFrame(
                {'sub_center_id': anm_mask, 'scores': y_pred_mask[:, 1]})
            scores_df.to_csv('outputs/scores.csv')






# if __name__ == '__main__':
#     hdps = HDPS()
#     test_fraud_prob, history_vect = hdps.preprocess_data()
#     hdps.predict_scores_next(history_vect)

