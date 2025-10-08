# Dataprocessing

In this section,the data has been collected from APIs, from which metrics have been calculated and then the data has been normalised using a rolling z score, which has been uplodaded to a database. This can be used in the mlmodel section to train the model. We have about 10 years/4000 tokens of data collected to train the model. 

# Core Market Data

In this file, data was retrieved from the yahoo finance API from which metrics and technical indicators could be derived. the mathematical theory behind the calculation of these values is covered in the analysis section. The dataframe was uploaded to a csv file which was used to store the data in the short term before the database could be created. 

sources:
https://pandas.pydata.org/
