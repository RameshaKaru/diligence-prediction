import time
from .hdps import HDPS
from .norm_score import NormScore

if __name__ == '__main__':
    start_time = time.time()

    hdps = HDPS()

    # preprocess the data
    test_fraud_prob, history_vect, meta_features = hdps.preprocess_data()

    # predicts the fcm scores for the next 4 weeks
    hdps.predict_scores_next(history_vect)
    # predicts the norm scores for the next 4 weeks and gets the intersection of lowest tier ANMs from both scores
    norms = NormScore(hdps)
    norms.predict_norm_scores_next(history_vect)
    norms.categorize_ANM()

    #gets the actual fcm and norm scores of the last 6 months and saves them to csvs
    hdps.get_past_scores(test_fraud_prob, meta_features)
    norms.get_past_scores(test_fraud_prob, meta_features)

    print("end")
    print("--- %s seconds ---" % (time.time() - start_time))