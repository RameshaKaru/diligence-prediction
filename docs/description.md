### General description of the implementation

####Basic steps done in the code
- Find percentages according to the 18 rules and draw KDEs
- Calculate the diligence probabilities using the KDE
- Cluster them to diligence and non-diligence clusters using c-means
- Find the c-means centers of 2 clusters
- Label data accordingly (not a hard classification)
- Train the model
- Predict the non-diligence score using history

####Fixed steps
- KDEs are drawn using the previously provided data set. They will not be recalculated
- Cluster centers which are found using c-means algorithm are fixed and will not be recalculated.
- Model will not be retrained on new data

####Steps for new data
- Provide the data in excel format
- Add the location of data, file names, date ranges and ANM sub center ids which need scores to the config file 

- Code calculates the diligence vectors using the KDEs drawn for 18 rules
- And predicts the non-diligence scores for the next 4 weeks, using history of 6 months of non-diligence vectors




