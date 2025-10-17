# Dataprocessing

In this section,the data has been collected from APIs, from which metrics have been calculated and then the data has been normalised using a rolling z score, which has been uplodaded to a database. This can be used in the mlmodel section to train the model. We have about 10 years/4000 tokens of data collected to train the model. 

# Core Market Data

In this file, data was retrieved from the yahoo finance API from which metrics and technical indicators could be derived. the mathematical theory behind the calculation of these values is covered in the analysis section. The dataframe was uploaded to a csv file which was used to store the data in the short term before the database could be created. 

# On chain metrics

In this file, on chain metrics have been retrieved from Google BigQuery. While Active addressess, Hash rate, Transaction Volume and Circulating supply were easy to find, Exchange flows posed a greater challenge. Upon researching into different APIs I found that the only way to calculate exchange flows was by analysing blockchain data myself (inaccurate and also extremely time consuming for 2 metrics) or paying upwards of $100 for some service. Google Big Query is a comprehensive database part of the google cloud service which comes with $300 of free credits, and while it did not have exchange flows as a standalone metric we could derive them fairly easily from other metrics which the flows correlate with. The process of calculating these metrics is contained within the estimate_exchange_flows_from_patterns function of the CompleteDailyBitcoinMetrics class, and is mathematically fairly simple. Accuracy improves for later tokens as the metric is based on more recent correlations and given that we will only be able to use data which is not null in core market data, on chain metrics or derivatives older data cannot actually be used even if we have data going back to early 2009, since core market data only goes back to 2015. 

Within some of these functions we have had to query the database by using SQL statements in order to retrieve the data. Some operations like SUM and AVG have been performed within the SQL for simplicity. We can then take the retrieved data from the SQL statement and put it into a Pandas dataframe, which can then easily be ported to the CSV file. 


# Derivatives

In this file, the derivatives metrics have been retrieved and calculated from Derbit. This was by far the longest and most complex API retrival yet, and took a long time given the constraints around rate limits, but in around 500 lines of code we managed to get futures, options and order book data going back to May 2019 into a CSV file. To retrieve data we had to convert dates into timestamps to match the arguments listed in the docs, and then send the API request using the requests module. With this final CSV file we can now create the data tables for derivatives, on chain metrics and core market data as well as a final dataset for the ML model with normalised values which can be passed in to the network. 
# Sources
https://pandas.pydata.org/
https://console.cloud.google.com/bigquery
https://www.cryptodatadownload.com/data/deribit/
https://www.deribit.com/statistics/BTC/options-data
https://docs.deribit.com/

