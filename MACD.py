def calculateMACD (df, long = 26, short = 12, lSignal = 9):
    """
    Calculates the moving average convergence divergence values for the dataframe

    Inputs:
        df (DataFrame) - dataframe that contains the data of the equity of interest
        long (int) - the length of longer EMA (general metric = 26)
        short (int) - the length of the shofter EMA (general metric = 12)
        signal (int) - the timeframe to compute the signal for the MACD (general metric = 9)
    Outputs:
        macd (Series) - a column containing macd for the corresponding entry
        signal (Series) - a column containing the signal for the corresponding entry
    """
    shortma = df["Close"].ewm(span = short, adjust = False).mean()
    longma = df["Close"].ewm(span = long, adjust = False).mean()

    macd = shortma - longma
    signal = macd.ewm(span = lSignal, adjust = False).mean()

    MACDcrossover = []

    for i in range(len(macd)):
        macd1 = macd[i-1]
        signal1 = signal[i-1]
        macd2 = macd[i]
        signal2 = signal[i]
        if macd1 < signal1 and macd2 > signal2:
            MACDcrossover.append("Buy")
        elif macd1 > signal1 and macd2 < signal2:
            MACDcrossover.append("Sell")
        else:
            MACDcrossover.append("Neutral")

    return macd, signal, MACDcrossover

def addMACD(df):
    """
    Adds all MACD columns to the dataframe for general analysis of MACD indicator
    Inputs:
        df (DataFrame) - dataframe to add columns to 
    Outputs:
        data (DataFrame) - dataframe with added columns
    
    """
    data = df.copy()
    macd, signal, crossover = calculateMACD(data)
    data["MACD"] = macd
    data["MACDSignal"] = signal
    data["MACDCrossover"] = crossover
    return data

def addKeyMACD(df):
    """
    Adds only significant MACD columns to the dataframe for use in regression model
    Inputs:
        df (DataFrame) - dataframe to add column to 
    Outputs:
        data (DataFrame) - dataframe with added column
    
    """
    data = df.copy()
    macd, signal, crossover = calculateMACD(data)
    data["MACD"] = macd
    return data