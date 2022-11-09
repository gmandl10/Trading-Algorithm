def calculateCCI(df, period = 20):
    """
    Calculates the Commodity Channel Index values for the dataframe

    Inputs:
        df (DataFrame) - dataframe that contains the data of the equity of interest
        period (int) - period to calculate the indidcator on, standard value is 20
        
    Outputs:
        cci (Series) - a column containing cci values for the corresponding entry

    """   
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    typical = (high + low + close)/3

    MA = typical.rolling(period).mean()

    mean_deviation = abs(typical - MA)

    cci = (typical -MA)/0.015*mean_deviation

    CCI_indicator = []
    oversold = -150
    overbought = 150

    CCI_indicator.append("Neutral")

    i = 1
    while i < len(cci):
        CCI1 = cci[i-1]
        CCI2 = cci[i]
        if CCI1 > oversold and CCI2 < oversold:
            CCI_indicator.append("Buy")
        elif CCI1 < overbought and CCI2 > overbought:
            CCI_indicator.append("Sell")
        else:
            CCI_indicator.append("Neutral")
        i+=1

    return cci, CCI_indicator

def addCCI(df):
    """
    Adds all Commodity Channel Index columns to the dataframe for general analysis
     of Commodity Channel Index
    Inputs:
        df (DataFrame) - dataframe to add columns to 
    Outputs:
        data (DataFrame) - dataframe with added columns
    
    """
    data = df.copy()
    cci, ccii = calculateCCI(data)
    data["CCI"] = cci
    data["CCIIndicator"] = ccii
    return data

def addKeyCCI(df):
    """
    Adds only significant Commodity Channel Index columns to the dataframe for use in regression model
    Inputs:
        df (DataFrame) - dataframe to add column to 
    Outputs:
        data (DataFrame) - dataframe with added column
    
    """
    data = df.copy()
    cci, ccii = calculateCCI(data)
    data["CCI"] = cci
    return data
