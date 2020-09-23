import time
from .hdps import HDPS

if __name__ == '__main__':
    start_time = time.time()

    hdps = HDPS()
    test_fraud_prob, history_vect = hdps.preprocess_data()
    hdps.predict_scores_next(history_vect)

    print("end")
    print("--- %s seconds ---" % (time.time() - start_time))