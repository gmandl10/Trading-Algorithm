
import pandas as pd


def calculateKeltnerChannel(df, period = 14):
    """
    Calculates the Keltner Channel values for the dataframe

    Inputs:
        df (DataFrame) - dataframe that contains the data of the equity of interest
        period (int) - period to calculate the indidcator on, standard value is 14
    Outputs:
        upperband (Series) - a column containing the upper band values for the
        corresponding entry
        lowerband (Series) - a column containing the lower band values for the
        corresponding entry
    """

    ema = df["Close"].ewm(span = period, adjust = False).mean()

    t1 = pd.DataFrame(df["High"] - df["Low"])
    t2 = pd.DataFrame(abs(df["High"] - df["Close"].shift(1)))
    t3 = pd.DataFrame(abs(df["Low"] - df["Close"].shift(1)))

    tr = pd.concat([t1, t2, t3], axis = 1, join = 'inner').max(axis = 1)
    atr = tr.rolling(period).mean()

    upperband = ema + 2*atr
    lowerband = ema - 2*atr
    keltner_indicator = []
    keltner_indicator.append("Neutral")
    i=0

    while i < len(upperband)-1:
        upper = upperband[i] 
        lower = lowerband[i]
        p1 = df["Close"][i]
        p2 = df["Close"][i+1]
        if p1 < lower and p1 < p2:
            keltner_indicator.append("Buy")
        elif p1 > upper and p1 > p2:
            keltner_indicator.append("Sell")
        else:
            keltner_indicator.append("Neutral")
        i+=1
    
    return lowerband, upperband, keltner_indicator

def addKeltnerChannel(df):
    """
    Adds all Keltner Channel columns to the dataframe for general analysis of Keltner Channel indicator
    Inputs:
        df (DataFrame) - dataframe to add columns to 
    Outputs:
        data (DataFrame) - dataframe with added columns
    
    """
    data = df.copy()
    lower, upper, ind= calculateKeltnerChannel(data)
    data["KeltnerUpperBand"] = lower
    data["KeltnerLowerBand"] = upper
    data["KeltnerIndicator"] = ind
    return data

def addKeyKeltnerChannel(df):
    """
    Adds only significant Keltner Channel columns to the dataframe for use in regression model
    Inputs:
        df (DataFrame) - dataframe to add columns to 
    Outputs:
        data (DataFrame) - dataframe with added columns
    
    """
    data = df.copy()
    lower, upper, ind= calculateKeltnerChannel(data)
    data["KeltnerRange"] = upper - lower
    return data