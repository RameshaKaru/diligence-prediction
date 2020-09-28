## How to get the non-diligence scores prediction for the next time window?

1. Setup your environment following the [setup guide](../setup/README.md)
2. Add the data files (Excel files) to the /data folder (or a folder you prefer)
3. Add the configuration file in the [provided format](../config.yaml) to the root folder.
(Details are available [here](../data/README.md))
4. Execute the below command to get the predicted non-diligence scores from within the root project folder.

    ```commandline
    python -m diligenceprediction
    ```

5. The predicted non diligence scores of the ANMs will be available in a csv file (scores.csv) in /outputs directory.
    - ANMs with higher scores are more non-diligent
    - These non-diligence scores corresponds to the prediction for the upcoming 4 weeks, from the last processed time window of the data set
    
6. Additionally, the actual scores of the past 6 months (calculated using the cmeans cluster centers) will be available in the /outputs directory/past_scores{i}.csv

> Main documentation [link](../docs/README.md)
