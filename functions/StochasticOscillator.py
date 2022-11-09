def calculateStochasticOscillator (df, period = 14, signal = 3):
    """
    Calculates the Stochastic Oscillator values for the dataframe

    Inputs:
        df (DataFrame) - dataframe that contains the data of the equity of interest
        period (int) - period to calculate the indidcator on, standard value is 14
        signal (int) - the timeframe to compute the signal for the Stochastic Oscillator,
         standard value is 3
        
    Outputs:
        so (Series) - a column containing Stochastic Oscillator for the corresponding entry
        sosignal (Series) - a column containing Stochastic Oscillator signal for the 
            corresponding entry
    """
    highest_high = df["High"].rolling(period).max()
    lowest_low = df["Low"].rolling(period).min()
    
    so = 100*((df["Close"] - lowest_low)/ (highest_high - lowest_low))
    sosignal = so.rolling(signal).mean()

    stochastic_indicator = []
    for i in range(len(so)):
        s = so[i]
        ma = sosignal[i]
        if s < 20 and ma < 20 and s < ma:
            stochastic_indicator.append("Buy")
        elif s > 80 and ma > 80 and s > ma:
            stochastic_indicator.append("Sell")
        else:
            stochastic_indicator.append("Neutral")

    return so, sosignal, stochastic_indicator

def addStochasticOscillator(df):
    """
    Adds all Stochastic Oscillator columns to the dataframe for general analysis
    of Stochastic Oscillator indicator
    Inputs:
        df (DataFrame) - dataframe to add columns to 
    Outputs:
        data (DataFrame) - dataframe with added columns
    
    """
    data = df.copy()
    so, sos, soi = calculateStochasticOscillator(data)
    data["StochasticOscillator"] = so
    data["SOSignal"] = sos
    data["SOIndicator"] = soi
    return data

def addKeyStochasticOscillator(df):
    """
    Adds only significant Stochastic Oscillator columns to the dataframe for use in regression model
    Inputs:
        df (DataFrame) - dataframe to add column to 
    Outputs:
        data (DataFrame) - dataframe with added column
    
    """
    data = df.copy()
    so, sos, soi = calculateStochasticOscillator(data)
    data["StochasticOscillator"] = so
    return data

