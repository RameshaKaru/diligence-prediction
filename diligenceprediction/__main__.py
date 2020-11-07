import time
from .hdps import HDPS
from .norm_score import NormScore

if __name__ == '__main__':
    start_time = time.time()

    hdps = HDPS()

    # preprocess the data
    test_fraud_prob, history_vect, meta_features = hdps.preprocess_data()

    # predicts the scores for the next 4 weeks
    hdps.predict_scores_next(history_vect)

    # gets the actual scores of the last 6 months and saves them to csvs
    hdps.get_past_scores(test_fraud_prob, meta_features)

    norms = NormScore(hdps)
    norms.predict_norm_scores_next(history_vect)
    norms.get_intersection()

    print("end")
    print("--- %s seconds ---" % (time.time() - start_time))