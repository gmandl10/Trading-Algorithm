def calculateWilliamsR(df, period = 14):
    """
    Calculates the Williams %R values for the dataframe
    Inputs:
        df (DataFrame) - dataframe that contains the data of the equity of interest
        period (int) - period to calculate the indidcator on, standard value is 14
    Outputs:
        williamsr (Series) - a column containing Williams %R indicator for the corresponding entry
    """
    highest_high = df["High"].rolling(period).max()
    lowest_low = df["Low"].rolling(period).min()
    williamsr = -100 * ((highest_high - df["Close"])/(highest_high-lowest_low))
    wr_indicator = []
    wr_indicator.append("Neutral")
    i=1
    while i < len(williamsr):
        wr1 = williamsr[i-1] 
        wr2 = williamsr[i]
        if wr1 > -80 and wr2 < -80:
            wr_indicator.append("Buy")
        elif wr1 < -20 and wr2 > -20:
            wr_indicator.append("Sell")
        else:
            wr_indicator.append("Neutral")

        i+=1
    return williamsr, wr_indicator

def addWilliamsR(df):
    """
    Adds all Williams %R columns to the dataframe for general analysis of Williams %R indicator
    Inputs:
        df (DataFrame) - dataframe to add columns to 
    Outputs:
        data (DataFrame) - dataframe with added columns
    
    """
    data = df.copy()
    williamsr, williamsrsignal = calculateWilliamsR(data)
    data["Williams%R"] = williamsr
    data["WRSignal"] = williamsrsignal
    return data

def addKeyWilliamsR(df):
    """
    Adds only significant Williams %R columns to the dataframe for use in regression model
    Inputs:
        df (DataFrame) - dataframe to add column to 
    Outputs:
        data (DataFrame) - dataframe with added column
    
    """
    data = df.copy()
    williamsr, williamsrsignal = calculateWilliamsR(data)
    data["Williams%R"] = williamsr
    return data
