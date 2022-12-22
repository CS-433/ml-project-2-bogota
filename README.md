# CS-433: Machine Learning Fall 2022, Project 2 
- Topic: *Machine learning in finance - Forecasting and Trading*
- Team Name: Bogota
- Team Member:
    1. Lelièvre Maxime, **SCIPER: 296777** (maxime.lelievre@epfl.ch)
    2. Peduto Matteo, **SCIPER: 316194** (matteo.peduto@epfl.ch)
    3. Mery Tom, **SCIPER: 297217** (tom.mery@epfl.ch)

* [Getting started](#getting-started)
    * [Project description](#project-description)
    * [Data](#data)
    * [Report](#report)
* [Reproduce results](#reproduce-results)
    * [Requirements](#Requirements)
    * [Repo Architecture](#repo-architecture)
    * [Instructions to run](#instructions-to-run)
* [Results](#results)

# Getting started
This repository contains the codes used to produce the results presented in `report.pdf`
## Project description

Over the last few years, the cryptocurrencies became in-creasingly popular as an investment product and for a portfolio
diversification strategy. A still increasing
body of literature focused on the pertinence of the efficient
market hypothesis (EMH). In essence, the EMH
postulates that efficient markets reflect all past, public or public
and private information in market prices. Verification of the
EMH is important for market participants as it implies that
such information cannot be used to make persistent profits
on trading on the market. 

In this context, we propose, in the continuation of Wildi
et al.(2019), to extend their approachs to other cryptocurrencies
and to commodities and market indices to see weather positive
trading performances can be achieved with forecasting using machine learning
models.

We here implement the following four machine learning methods to forecast the log-returns of several assets:

- Feedforward Neural Network (NN)
- Convolutional Neural Network (CNN)
- Long-Short-Term-Memory (LSTM)
- Random Forest


We further analyze weather the combination of several models
trained on one asset gives better results than the results of
the best model only and weather the combination of the same
model trained on different assets of the same type gives better results than the results of the model trained on only one asset of this type. Therefore
we test their performance with several trading performance
metrics.

## Data
The collection of data has been performed on different web sites herafter detailed with the assets’ symbol used in the code:

| Asset type | Asset name | Symbol | Periods | Link |
| -----------| ---------- | ------ | -------------------- |-|
| Crypto-currency| Bitcoin | BTC-USD | 2017-11-09 to 2022-12-13 | [here](https://www.cryptodatadownload.com/data/bitstamp/) 
| Crypto-currency| Ether | ETH-USD | 2017-11-09 to 2022-12-13 | [here](https://www.cryptodatadownload.com/data/bitstamp/) 
| Crypto-currency| Ripple | XRP-USD | 2017-11-09 to 2022-12-13 | [here](https://www.cryptodatadownload.com/data/bitstamp/) 
| Commodity| Gold | LBMA-GOLD | 2012-12-13 to 2022-12-09 | [here](https://data.nasdaq.com/data/LBMA/GOLD-gold-price-london-fixing) 
| Commodity| Natural Gas | NYMEX-NG | 2012-12-13 to 2022-12-09 | [here](https://www.nasdaq.com/market-activity) 
| Commodity| Oil | OPEC-ORB | 2012-12-13 to 2022-12-09 | [here](https://data.nasdaq.com/data/OPEC/ORB-opec-crude-oil-price) 
| Stock market index| S&P500 | SP500 | 2012-12-13 to 2022-12-12| [here](https://www.nasdaq.com/market-activity) 
| Stock market index| SMI | SMI | 2012-12-13 to 2022-12-12| [here](https://finance.yahoo.com/) 
| Stock market index| CAC40 | CAC40 | 2012-12-13 to 2022-12-12| [here](https://finance.yahoo.com/) 

The raw data are already available in the data folder.
## Report
All the details about the choices that have been made and the methodology used throughout this project are available in `report.pdf`. Through this report, the reader is able to understand the different assumptions, decisions and results made during the project. The theoretical background is also explained.
# Reproduce results
## Requirements
If you are using conda you can directly run the following to install all the requirements:
```
conda env create -f environment.yml
conda activate ml_project_finance
```
Otherwise the dependencies are available in `environment.yml`. If you also want to run the notebooks, after activating `ml_project_finance` execute:
```
conda install ipykernel
```

## Repo Architecture
<pre>  
├─── bestmodels
    ├─── commodities
        ├─── LBMA-GOLD
            ├─── CNN.pkl
            ├─── LSTM.pkl
            ├─── NN.pkl
            ├─── params_CNN.pkl
            ├─── params_LSTM.pkl
            ├─── params_NN.pkl
            ├─── params_RandomForest.pkl
        ├─── NYMEX-NG
            ├─── CNN.pkl
            ├─── LSTM.pkl
            ├─── NN.pkl
            ├─── params_CNN.pkl
            ├─── params_LSTM.pkl
            ├─── params_NN.pkl
            ├─── params_RandomForest.pkl
        ├─── OPEC-ORB
            ├─── CNN.pkl
            ├─── LSTM.pkl
            ├─── NN.pkl
            ├─── params_CNN.pkl
            ├─── params_LSTM.pkl
            ├─── params_NN.pkl
            ├─── params_RandomForest.pkl
    ├─── cryptos
        ├─── BTC-USD
            ├─── CNN.pkl
            ├─── LSTM.pkl
            ├─── NN.pkl
            ├─── params_CNN.pkl
            ├─── params_LSTM.pkl
            ├─── params_NN.pkl
            ├─── params_RandomForest.pkl
        ├─── ETH-USD
            ├─── CNN.pkl
            ├─── LSTM.pkl
            ├─── NN.pkl
            ├─── params_CNN.pkl
            ├─── params_LSTM.pkl
            ├─── params_NN.pkl
            ├─── params_RandomForest.pkl
        ├─── XRP-USD
            ├─── CNN.pkl
            ├─── LSTM.pkl
            ├─── NN.pkl
            ├─── params_CNN.pkl
            ├─── params_LSTM.pkl
            ├─── params_NN.pkl
            ├─── params_RandomForest.pkl
    ├─── stock_market_index
        ├─── CAC40
            ├─── CNN.pkl
            ├─── LSTM.pkl
            ├─── NN.pkl
            ├─── params_CNN.pkl
            ├─── params_LSTM.pkl
            ├─── params_NN.pkl
            ├─── params_RandomForest.pkl
        ├─── SMI
            ├─── CNN.pkl
            ├─── LSTM.pkl
            ├─── NN.pkl
            ├─── params_CNN.pkl
            ├─── params_LSTM.pkl
            ├─── params_NN.pkl
            ├─── params_RandomForest.pkl
        ├─── SP500
            ├─── CNN.pkl
            ├─── LSTM.pkl
            ├─── NN.pkl
            ├─── params_CNN.pkl
            ├─── params_LSTM.pkl
            ├─── params_NN.pkl
            ├─── params_RandomForest.pkl
├─── data
    |─── processed
        ├─── commodities
            ├─── LBMA-GOLD
                ├─── test_data.pkl
                ├─── train_data.pkl
                ├─── val_data.pkl
            ├─── NYMEX-NG
                ├─── test_data.pkl
                ├─── train_data.pkl
                ├─── val_data.pkl
            ├─── OPEC-ORB
                ├─── test_data.pkl
                ├─── train_data.pkl
                ├─── val_data.pkl
        ├─── cryptos
            ├─── BTC-USD
            ├─── ETH-USD
            ├─── XRP-USD
        ├─── stock_market_index
            ├─── CAC40
            ├─── SMI
            ├─── SP500
    |─── raw
        ├─── commodities
            ├─── LBMA-GOLD.csv
            ├─── NYMEX-NG.csv
            ├─── OPEC-ORB.csv
        ├─── cryptos
            ├─── BTC-USD.csv
            ├─── ETH-USD.csv
            ├─── XRP-USD.csv
        ├─── stock_market_index
            ├─── CAC40.csv
            ├─── SMI.csv
            ├─── SP500.csv
├─── figures
    ├─── 
├─── notebooks
    ├─── data_analysis.ipynb: Exploratory data analysis notebooks.
├─── README.md: README
├─── references
    ├─── bitcoin_and_market_inefficiency.pdf
    ├─── project2_description.pdf
├─── src
    ├─── __init__.py: File to define src directory as a python package
    ├─── models.py
    ├─── process_data.py
    ├─── test.py
    ├─── trading_utils.py
    ├─── train.py
    ├─── validation.py
├─── report.pdf: Report explaining methods and choices that have been made.
</pre>

## Instructions to run 
The following commands give more details about the positional arguments and a description of the process done while running:

```
python process_data.py -h
python validation.py -h
python train.py -h
python test.py -h
```
Please run them before running the following. The commands showed bellow have to be executed in the same order to keep consistency.

The processed data can be reproduced from the raw data by moving to the `src/` folder and execute:
```
python process_data.py dataset nb_lags train_ratio
````

To run the optimization on the validation set move to the `src/` folder and execute:
```
python validation.py model_type dataset
```
Beware that optimizing one model type on one dataset takes from 1min to 8 min (depending on the model) on Google Colab with GPU availability.

To train the models with the best parameters found during the validation move to the `src/` folder and execute:
```
python train.py model_type dataset
```

To test the performances of the trained models move to the `src/` folder and execute:
```
python test.py model_type dataset
````
# Results
