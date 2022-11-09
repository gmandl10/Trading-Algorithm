def calculateAroon(df, period = 25):
    """
    Calculates the Aroon Indicator values for the dataframe

    Inputs:
        df (DataFrame) - dataframe that contains the data of the equity of interest
        period (int) - period to calculate the indidcator on, standard value is 25
        
    Outputs:
        aroonup (Series) - a column containing Aroon Up values for the corresponding entry
        aroondown (Series) - a column containing Aroon Down values for the corresponding entry
    """   
    high = df["High"]
    low = df["Low"]

    aroonup = 100 * high.rolling(period + 1).apply(lambda x: x.argmax()) / period
    aroondown = 100 * low.rolling(period + 1).apply(lambda x: x.argmin()) / period

    aroon_crossover = []
    aroon_crossover.append("Neutral")

    i= 1
    while i < len(aroonup):
        aroonup1 = aroonup[i-1]
        aroonup2 = aroonup[i]
        aroondown1 = aroondown[i-1]
        aroondown2 = aroondown[i]
        if aroonup1 < aroondown1 and aroonup2 > aroondown2:
            aroon_crossover.append("Long")
        elif aroonup1 > aroondown1 and aroonup2 < aroondown2:
            aroon_crossover.append("Short")
        else:
            aroon_crossover.append("Neutral")
        
        i += 1

    aroon_indicator = []

    for i in range(len(aroonup)):
        up = aroonup[i]
        down = aroondown[i]

        if up > 70 and down < 30:
            aroon_indicator.append("Long")
        elif down > 70 and up < 30: 
            aroon_indicator.append("Short")
        elif down < 50 and up < 50:
            aroon_indicator.append("PriceConsolidating")
        else:
            aroon_indicator.append("Neutral")
    
    return aroonup, aroondown, aroon_crossover, aroon_indicator


def addAroon(df):
    """
    Adds all Aroon Indicator columns to the dataframe for general analysis of Aroon Indicator
    Inputs:
        df (DataFrame) - dataframe to add columns to 
    Outputs:
        data (DataFrame) - dataframe with added columns
    
    """
    data = df.copy()
    u, d, ac, ai = calculateAroon(data)
    data["AroonUp"] = u
    data["AroonDown"] = d
    data["AroonCrossover"] = ac
    data["AroonIndicator"] = ai
    return data

def addKeyAroon(df):
    """
    Adds only significant Aroon Indicator columns to the dataframe for use in regression model
    Inputs:
        df (DataFrame) - dataframe to add column to 
    Outputs:
        data (DataFrame) - dataframe with added column
    
    """
    data = df.copy()
    u, d, ac, ai = calculateAroon(data)
    data["AroonUp"] = u
    data["AroonDown"] = d
    return data
