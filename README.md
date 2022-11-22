Hello

# Step 1: Reproduce results of Marc on neural nets

- **Data**: historical data of the Bitcoin (BTC) price expressed in USD daily for the period of April 15, 2014 to Januar 10, 2019. Data from yahoo start at Sep 17, 2014, Take Adj Close as P(t) (price at time t) 
- **Model**: Feedforward fully connected neural nets, 
- **Input**: the first six lags of the Bitcoin log returns r(t-i) = (log(P(t-i)/P(t-i-1))), i = 0,1,2,3,4,5
- **Architecture**: 
    - Input layer: dim 6
    - Hidden layer 1: dim 6
    - Hidden layer 2: dim 3 
    - Ouput layer: dim 1 
    - Activation: sigmoid
- **Output**: future log return, r(t+1)
- **Data splitting**: Training and validation samples were roughly split into 1/2 and 1/2 of the dataset so that the in-sample period ends in 2016-06-01.
- **Loss metric**: MSE to minimize
- **Trading strategy**: if forecasted r(t+1) > 0, buy at time (t) and sell at time (t+1)
- **Performance metrics**: 
    - Hit rate (should be maximized): percentage of positions that have generated positive returns.
    - Max draw down (should be minimized): the maximum observed loss from a peak to a trough. 
    - Annualized sharpe: the product of the monthly Sharpe Ratio and the square root of 12 (see [here](https://awgmain.morningstar.com/webhelp/glossary_definitions/mutual_fund/mfglossary_Sharpe_Ratio.html#:~:text=The%20annualized%20Sharpe%20Ratio%20is,12%20(annualized%20standard%20deviation).)).
- **Expected performance** (over 100 random neural nets): 
    - Hit rate: 0.679
    - Max draw down: 0.558
    - Annualized sharpe: 1.639

# Step 2: Improve performance using new methods
Maybe we could try:
- Larger prediction window
- Deeper and/or wider Neural nets
- New activation functions

And other model:
- Random forest
- CNN
- RNN
- LSTM