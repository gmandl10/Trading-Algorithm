def calculateBollingerBand (df, period = 14, factor = 2):
    """
    Calculates the bollinger band values for the dataframe

    Inputs:
        df (DataFrame) - dataframe that contains the data of the equity of interest
        period (int) - period to calculate the indidcator on, standard value is 14
        factor (int) - the factor to multiply the standard deviations by
    Outputs:
        upperbb (Series) - a column containing the upper bollinger band values for the
        corresponding entry
        lowerbb (Series) - a column containing the lower bollinger band values for the
        corresponding entry
        sma (Series) - a column containing the SMA ofr the corresponding entry
    """
    price = df["Close"]
    sma = price.rolling(period).mean()
    sd = price.rolling(period).std()

    upperbb = sma + sd*factor
    lowerbb = sma - sd*factor

    bollingerindicator = []
    for i in range(period):
        bollingerindicator.append("None")
    for i in range(14, len(price)):
        percentile = (price[i] - lowerbb[i])/(upperbb[i] - lowerbb[i])
        if percentile < 0.05:
            bollingerindicator.append("Strongly Oversold")
        elif percentile < 0.25:
            bollingerindicator.append("Oversold")
        elif percentile > 0.95:
            bollingerindicator.append("Strongly Overbought")
        elif percentile > 0.75:
            bollingerindicator.append("Overbought")
        else:
            bollingerindicator.append("None")
    return upperbb, lowerbb, sma, bollingerindicator

def addBollingerBand(df):
    """
    Adds all Bollinger Band columns to the dataframe for general analysis of Bollinger Band indicator
    Inputs:
        df (DataFrame) - dataframe to add columns to 
    Outputs:
        data (DataFrame) - dataframe with added columns
    
    """
    data = df.copy()
    upperbb, lowerbb, sma, bi= calculateBollingerBand(data)
    
    data["LowerBB"] = lowerbb
    data["SMA"] = sma
    data["UpperBB"] = upperbb
    data["BollingerIndication"] = bi
    return data

def addKeyBollingerBand(df):
    """
    Adds only significant Bollinger Band columns to the dataframe for use in regression model
    Inputs:
        df (DataFrame) - dataframe to add column to 
    Outputs:
        data (DataFrame) - dataframe with added column
    
    """
    data = df.copy()
    upperbb, lowerbb, sma = calculateBollingerBand(data)
    
    data["BollingerRange"] = upperbb - lowerbb
    return data

