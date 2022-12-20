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
All the detailes about the choices that have been made and the methodology used throughout this project are available in `report.pdf`. Through this report, the reader is able to understand the different assumptions, decisions and results made during the project. The theoretical background is also explained.
# Reproduce results
## Requirements
- Python==3.9.13
- Numpy==1.21.5
- Matplotlib

## Repo Architecture
<pre>  
├─── bestmodels
    ├─── commodities
        ├─── LBMA-GOLD
        ├─── NYMEX-NG
        ├─── OPEC-ORB
    ├─── cryptos
        ├─── BTC-USD
        ├─── ETH-USD
        ├─── XRP-USD
    ├─── stock_market_index
        ├─── CAC40
        ├─── SMI
        ├─── SP500
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
Move to the root folder and execute:

    python run.py

Make sure to have all the requirements and the data folder in the root. Be aware training the models on 1000 epochs takes around 5 min on Apple silicon M1 Pro. Here the best model has been trained over 15000 epochs.

If you want to run the cross-validation move to the root folder and execute:

    python optimization.py

Here the cross-validation has taken around 1h for one sub-models (on Apple silicon M1 Pro), therefore around 8 hours for the whole model.

If you want to visualize the performances of the model during the training, move to the root folder and execute:

    python plot_performance.py

# Results
The performances of the models is assessed on AirCrowd from `data/submission.csv` generated by `run.py`. The model achieves a global accuracy of 0.818 with a F1-score of 0.722.

Here are he performance of each sub model during the training:

[![IMAGE ALT TEXT HERE](https://github.com/CS-433/ml-project-1-los_caballeros_de_bogota/blob/main/figures/mass.jpeg)](https://github.com/CS-433/ml-project-1-los_caballeros_de_bogota/blob/main/figures)
[![IMAGE ALT TEXT HERE](https://github.com/CS-433/ml-project-1-los_caballeros_de_bogota/blob/main/figures/no_mass.jpeg)](https://github.com/CS-433/ml-project-1-los_caballeros_de_bogota/blob/main/figures)
