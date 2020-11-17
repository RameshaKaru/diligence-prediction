## How to get the non-diligence scores prediction for the next time window?

1. Setup your environment following the [setup guide](../setup/README.md)
2. Add the data files (Excel files) to the /data folder (or a folder you prefer)
3. Add the configuration file in the [provided format](../config.yaml) to the root folder.
(Details are available [here](../data/README.md))
4. Execute the below command to get the predicted non-diligence scores from within the root project folder.

    ```commandline
    python -m diligenceprediction
    ```

5. Non diligence scores of the ANMs will be available in a csv file (scores.csv) in /outputs directory.
    - ANMs with higher scores are more non-diligent
    - These non-diligence scores corresponds to the prediction for the upcoming 4 weeks, from the last processed time window of the data set
   
6. Additionally, the actual scores of the past 6 months (calculated using the cmeans cluster centers) will be available in _/outputs/past_scores{i}.csv_

7. Additional score - simple norm score will also be predicted for the same time window (i.e. upcoming 4 weeks). 
   The results will be available in _/outputs/norm_scores.csv_.

8. The ANMs in the lowest tier (bad performers) will be calculated considering the intersection of 20 worst performing ANMs in simple norm score and 20 worst performing ANMs in the fcm score. 
   The IDs of these ANMs will be written to _/outputs/low_tier_anms.csv_.
   (The parameter 20 can be adjusted by passing the desired value to get_intersection(x) function).
   
9. The actual simple norm scores of the past 6 months will be available in _/outputs/past_norm_scores{i}.csv_.
 
> Main documentation [link](../docs/README.md)
