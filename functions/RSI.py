import pandas as pd

def calculateRSI(df, period = 14):
    """
    Calculates the Relative Strength Index (RSI) values for the dataframe

    Inputs:
        df (DataFrame) - dataframe that contains the data of the equity of interest
        period (int) - period to calculate the indidcator on, standard value is 14
    Outputs:
        rsi (Series) - a column containing rsi for the corresponding entry
    """
    close = df["Close"]  
    diff = close.diff()
    gain = []
    loss = []
    for i in range(len(diff)):
        if diff[i] < 0:
            gain.append(0)
            loss.append(diff[i])
        else:
            gain.append(diff[i])
            loss.append(0)
    gain = pd.Series(gain)
    loss = pd.Series(loss)

    gainEMA = gain.ewm(span = period - 1, adjust = False).mean()
    lossEMA = abs(loss.ewm(span = period - 1, adjust = False).mean())
    rs = gainEMA/lossEMA
    rsi = 100 - (100 / (1 + rs))
    rsi_ = []
    for i in rsi:
        rsi_.append(i)
    rsisignal = []
    for i in range(len(rsi)):
        if rsi[i] < 30:
            rsisignal.append("Buy")
        elif rsi[i] > 70:
            rsisignal.append("Sell")
        else:
            rsisignal.append("Neutral")
    return rsi_, rsisignal
    
def addRSI(df):
    """
    Adds all RSI columns to the dataframe for general analysis of RSI indicator
    Inputs:
        df (DataFrame) - dataframe to add columns to 
    Outputs:
        data (DataFrame) - dataframe with added columns
    
    """
    data = df.copy()
    rsi, signal = calculateRSI(data)
    data["RSI"] = rsi
    data["RSISignal"] = signal
    return data

def addKeyRSI(df):
    """
    Adds only significant RSI columns to the dataframe for use in regression model
    Inputs:
        df (DataFrame) - dataframe to add column to 
    Outputs:
        data (DataFrame) - dataframe with added column
    
    """
    data = df.copy()
    rsi, s = calculateRSI(data)
    data["RSI"] = rsi
    return data
