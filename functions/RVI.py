def calculateRVI(df, period = 10):
    """
    Calculates the Relative Vigor Index (RVI) values for the dataframe

    Inputs:
        df (DataFrame) - dataframe that contains the data of the equity of interest
        period (int) - period to calculate the indidcator on, standard value is 10
        
    Outputs:
        rvi (Series) - a column containing RVI for the corresponding entry
        rvisignal (Series) - a column containing RVI signal for the corresponding entry
    """

    high = df["High"]
    low = df["Low"]
    _open = df["Open"]
    close = df["Close"]

    a = close - _open
    b = close.shift(1) - _open.shift(1)
    c = close.shift(2) - _open.shift(2)
    d = close.shift(3) - _open.shift(3)

    numerator = 1/6*(a+2*b+2*c+d)

    e = high - low
    f = high.shift(1) - low.shift(1)
    g = high.shift(2) - low.shift(2)
    h = high.shift(3) - low.shift(3)

    denominator = 1/6*(e + 2*f + 2*g +h)

    rvi = numerator.rolling(10).mean()/denominator.rolling(10).mean()

    i = rvi.shift(1)
    j = rvi.shift(2)
    k = rvi.shift(3)

    rvisignal = 1/6*(rvi+2*i+2*j+k)

    rvi_crossover = []
    rvi_crossover.append("Neutral")

    i = 1 
    while i < len(rvi):
        rvi1 = rvi[i-1]
        rvi2 = rvi[i]
        signal1 = rvisignal[i-1]
        signal2 = rvisignal[i]
        if rvi1 < signal1 and rvi2 > signal2:
            rvi_crossover.append("Buy")
        elif rvi1 > signal1 and rvi2 < signal2:
            rvi_crossover.append("Sell")
        else:
            rvi_crossover.append("Neutral")
        i+=1
    rvi_divergence = []
    rvi_divergence.append("Neutral")

    i = 1 
    while i < len(rvi):
        rvi1 = rvi[i-1]
        rvi2 = rvi[i]
        price1 = close[i-1]
        price2 = close[i]
        if price2 > price1 and rvi2 < rvi1:
            rvi_divergence.append("Sell")
        elif price1 > price2 and rvi1 < rvi2:
            rvi_divergence.append("Buy")
        else:
            rvi_divergence.append("Neutral")
        i+=1
    return rvi, rvisignal, rvi_crossover, rvi_divergence

def addRVI(df):
    """
    Adds all Relative Vigor Index columns to the dataframe for general analysis
    of Relative Vigor Index indicator
    Inputs:
        df (DataFrame) - dataframe to add columns to 
    Outputs:
        data (DataFrame) - dataframe with added columns
    
    """
    data = df.copy()
    rvi, rvis, rvic, rvid = calculateRVI(data)
    data["RVI"] = rvi
    data["RVISignal"] = rvis
    data["RVICrossover"] = rvic
    data["RVIDivergence"] = rvid
    return data

def addKeyRVI(df):
    """
    Adds only significant Relative Vigor Index columns to the dataframe for use in regression model
    Inputs:
        df (DataFrame) - dataframe to add column to 
    Outputs:
        data (DataFrame) - dataframe with added column
    
    """
    data = df.copy()
    rvi, rvis, crossover, divergence = calculateRVI(data)
    data["RVI"] = rvi
    return data
