import pandas as pd

def calculateADX (df, period = 14):
    """
    Calculates the Average Directional Index (ADX) values for the dataframe

    Inputs:
        df (DataFrame) - dataframe that contains the data of the equity of interest
        period (int) - period to calculate the indidcator on, standard value is 14
    Outputs:
        adx (Series) - a column containing adx for the corresponding date
        pdi (Series) - a column containing postive directional index for the corresponding entry
        ndi (Series) - a column containing negative directional index for the corresponding entry
    """
    plus_dm = df["High"].diff()
    minus_dm = df["Low"].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0

    t1 = pd.DataFrame(df["High"] - df["Low"])
    t2 = pd.DataFrame(abs(df["High"] - df["Close"].shift(1)))
    t3 = pd.DataFrame(abs(df["Low"] - df["Close"].shift(1)))

    tr = pd.concat([t1, t2, t3], axis = 1, join = 'inner').max(axis = 1)
    atr = tr.rolling(period).mean()

    pdi = 100 * (plus_dm.ewm(alpha = 1/period).mean() / atr)
    ndi = abs(100 * (minus_dm.ewm(alpha = 1/period).mean() / atr))
    dx = (abs(pdi - ndi) / abs(pdi + ndi)) * 100
    adx1 = ((dx.shift(1) * (period - 1)) + dx) / period
    adx = adx1.ewm(alpha = 1/period).mean()

    adx_indicator = []
    adx_indicator.append("Neutral")
    i=1

    while i < len(adx):
        adx1 = adx[i-1]
        adx2 = adx[i]

        if adx1 < 25 and adx2 > 25 and pdi[i] > ndi[i]:
            adx_indicator.append("Buy")
        elif adx1 < 25 and adx2 > 25 and ndi[i] > pdi[i]:
            adx_indicator.append("Sell")
        else:
            adx_indicator.append("Neutral")
        i+=1
        
    return adx, pdi, ndi, adx_indicator

def addADX(df):
    """
    Adds all ADX columns to the dataframe for general analysis of ADX indicator
    Inputs:
        df (DataFrame) - dataframe to add columns to 
    Outputs:
        data (DataFrame) - dataframe with added columns
    
    """
    data = df.copy()
    adx, pdi, ndi, adxindicator = calculateADX(data)
    data["ADX"] = adx
    data["PDI"] = pdi
    data["NDI"] = ndi
    data["ADXIndicator"] = adxindicator
    return data

def addKeyADX(df):
    """
    Adds only significant ADX columns to the dataframe for use in regression model
    Inputs:
        df (DataFrame) - dataframe to add column to 
    Outputs:
        data (DataFrame) - dataframe with added column
    
    """
    data = df.copy()
    adx, p, n, i= calculateADX(data)
    data["ADX"] = adx
    return data
