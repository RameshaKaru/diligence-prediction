from .hdps import HDPS
import pandas as pd
import numpy as np
from tensorflow import keras as keras


class NormScore:
    """
        This class predicts the simple norm scores for the next time window using the saved model
        Also allows obtaining the actual simple norm scores

    """

    def __init__(self, hdps):
        """
        During initialization:
        - gets the configurations from the config file
        - gets the KDEs
        - gets the saved model
        - gets the other required data

        """
        print("INITIALIZING NORM SCORE")

        self.norm_model = self.get_saved_norm_model()
        self.predictANMids = hdps.predictANMids
        self.allANMdf = hdps.allANMdf

    def get_saved_norm_model(self):
        """
        Obtains the saved norm model

        Returns
        -------
        keras model

        """
        norm_model = keras.models.load_model('models/norm_model')
        return norm_model


    def predict_norm_scores_next(self, history_vect):
        """
        Predicts the simple norm scores for the next time window and saves them in a csv file

        Parameters
        ----------
        history_vect : array
            history of non diligence probabilities

        """
        print("PREDICTING NORM SCORES")

        y_pred = self.norm_model.predict(history_vect)

        if self.predictANMids is None:
            print("IDs of the sub centers where camps will be held is not provided. Hence predicting for all")
            scores_df = pd.DataFrame(
                {'sub_center_id': np.array(self.allANMdf['sub_center_id']), 'norm_scores': y_pred[:, 0]})
            scores_df.to_csv('outputs/norm_scores.csv')
        else:
            next_anm_df = pd.DataFrame(
                {'sub_center_id': self.predictANMids, 'flag': np.full((len(self.predictANMids)), True)})
            df_merge = pd.merge(self.allANMdf, next_anm_df, how="left", on="sub_center_id")
            df_merge['flag'].fillna(False, inplace=True)
            mask = df_merge['flag']
            y_pred_mask = y_pred[mask]
            anm_mask = np.array(self.allANMdf['sub_center_id'])[mask]
            scores_df = pd.DataFrame(
                {'sub_center_id': anm_mask, 'norm_scores': y_pred_mask[:, 0]})
            scores_df.to_csv('outputs/norm_scores.csv')


    def get_intersection(self, tier_count=20):
        """
            Gets the non diligent ANMs using both cmeans score and simple norm score for the next time window and
            saves them in a csv file

            Parameters
            ----------
            tier_count : int
                how many worst performers are selected using each score

        """
        print("Getting non diligent ANMs using both scores")

        norm_df = pd.read_csv('outputs/norm_scores.csv')
        fcm_df = pd.read_csv('outputs/scores.csv')

        norm_ids = list(norm_df['norm_scores'])
        fcm_ids = list(fcm_df['scores'])

        norm_df = norm_df.sort_values(by=['norm_scores'], ascending=False, ignore_index=True)
        norm_ids = list(norm_df['sub_center_id'][:tier_count])

        fcm_df = fcm_df.sort_values(by=['scores'], ascending=False, ignore_index=True)
        fcm_ids = list(fcm_df['sub_center_id'][:tier_count])

        union_ids = list(set().union(norm_ids, fcm_ids))
        print("Number of ANMs in the low-performing tier: ", len(union_ids))

        low_tier_df = pd.DataFrame({'sub_center_id': union_ids})
        low_tier_df.to_csv('outputs/low_tier_anms.csv')

