def roc(price, n):
    """
    Calculates the Rate of Change (ROC) of a price over n periods
    """
    return (price.diff(n)/price.shift(n))*100

def calculateKST(df, signal = 9):
    """
    Calculates the Know Sure Thing (KST) values for the dataframe

    Inputs:
        df (DataFrame) - dataframe that contains the data of the equity of interest
        signal (int) - period to calculate the signal of indidcator on, standard value is 9
        
    Outputs:
        kst (Series) - a column containing KST for the corresponding entry
        kstsignal (Series) - a column containing KST signal for the corresponding entry
    """

    close = df["Close"]

    def roc(price, n):
        return (price.diff(n)/price.shift(n))*100
    r1 = roc(close, 10).rolling(10).mean()
    r2 = roc(close, 15).rolling(10).mean()
    r3 = roc(close, 20).rolling(10).mean()
    r4 = roc(close, 30).rolling(15).mean()
    kst = r1 + 2*r2 + 3*r3 + 4*r4

    kstsignal = kst.rolling(signal).mean()

    kst_crossover = []
    kst_crossover.append("Neutral")

    i = 1
    while i < len(kst):
        kst1 = kst[i-1]
        kst2 = kst[i]
        signal1 = kstsignal[i-1]
        signal2 = kstsignal[i]

        if kst1 < signal1 and kst2 > signal2:
            kst_crossover.append("Buy")
        elif kst1 > signal1 and kst2 < signal2:
            kst_crossover.append("Sell")
        else:
            kst_crossover.append("Neutral")
        i+=1

    return kst, kstsignal, kst_crossover

def addKST(df):
    """
    Adds all Know Sure Thing columns to the dataframe for general analysis
    of Know Sure Thing indicator
    Inputs:
        df (DataFrame) - dataframe to add columns to 
    Outputs:
        data (DataFrame) - dataframe with added columns
    
    """
    data = df.copy()
    kst, ksts, kstc = calculateKST(data)
    data["KST"] = kst
    data["KSTSignal"] = ksts
    data["KSTCrossover"] = kstc
    return data

def addKeyKST(df):
    """
    Adds only significant Know Sure Thing columns to the dataframe for use in regression model
    Inputs:
        df (DataFrame) - dataframe to add column to 
    Outputs:
        data (DataFrame) - dataframe with added column
    
    """
    data = df.copy()
    kst, ksts, crossover = calculateKST(data)
    data["KST"] = kst
    return data
