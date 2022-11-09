def calculateDisparity(df, period = 14):
    """
    Calculates the Disparity Index values for the dataframe

    Inputs:
        df (DataFrame) - dataframe that contains the data of the equity of interest
        period (int) - period to calculate the indidcator on, standard value is 14
        
    Outputs:
        disparity (Series) - a column containing disparity values for the corresponding entry
    """
    close = df["Close"]

    SMA = close.rolling(period).mean()
    disparity = (close - SMA)/(SMA*100)

    disparity_indicator = []

    for _ in range(5):
        disparity_indicator.append("Neutral")

    i = 5
    while i < len(disparity):
        di1 = disparity[i-5]
        di2 = disparity[i-4]
        di3 = disparity[i-3]
        di4 = disparity[i-2]
        di5 = disparity[i-1]
        ditoday = disparity[i]

        if di1 < 0 and di2 < 0 and di3 < 0 and di4 < 0 and di5 < 0 and ditoday > 0:
            disparity_indicator.append("Buy")
        elif di1 > 0 and di2 > 0 and di3 > 0 and di4 > 0 and di5 > 0 and ditoday < 0:
            disparity_indicator.append("Sell")
        else:
            disparity_indicator.append("Neutral")
        
        i+=1

    return disparity, disparity_indicator

def addDisparity(df):
    """
    Adds all Disparity Index columns to the dataframe for general analysis
    of Disparity Index indicator
    Inputs:
        df (DataFrame) - dataframe to add columns to 
    Outputs:
        data (DataFrame) - dataframe with added columns
    
    """
    data = df.copy()
    d, di = calculateDisparity(data)
    data["Disparity"] = d
    data["DisparityIndicator"] = di
    return data

def addKeyDisparity(df):
    """
    Adds only significant Disparity Index columns to the dataframe for use in regression model
    Inputs:
        df (DataFrame) - dataframe to add column to 
    Outputs:
        data (DataFrame) - dataframe with added column
    
    """
    data = df.copy()
    d, di = calculateDisparity(data)
    data["Disparity"] = d
    return data
    
