## General description of the implementation

#### Basic steps done in the code
- Find percentages according to the 18 rules and draw KDEs
- Calculate the diligence probabilities using the KDE
- Cluster them to diligence and non-diligence clusters using c-means
- Find the c-means centers of 2 clusters
- Label data accordingly (not a hard classification)
- Train the model
- Predict the 2 non-diligence scores (fcm & simple norm scores) using history
- Identify the worst performing ANMs

#### Fixed steps
- KDEs are drawn using the previously provided data set. They will not be recalculated
- Cluster centers which are found using c-means algorithm are fixed and will not be recalculated.
- Models will not be retrained on new data

#### Steps for new data
- Provide the data in excel format
- Add the location of data, file names, date ranges and ANM sub center ids which need scores to the config file 

- Code calculates the diligence vectors using the KDEs drawn for 18 rules
- And predicts 2 non-diligence scores (fcm score and simple norm score) for the next 4 weeks, using history of 6 months of non-diligence vectors
- Poor performing ANMs are identified from the intersection of worst _n_ performers in fcm scores and worst _n_ performers in simple norm scores. (n=20)

> Main documentation [link](README.md)


