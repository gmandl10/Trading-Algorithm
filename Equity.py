import yfinance as yf
import math
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#from sklearn import linear_model as lm
import datetime as dt

class Equity:


    def __init__(self, tickerSymbol, periods) -> None:
        self.symbol = tickerSymbol
        self.__setTicker()
        self.__setData()
        self.__setVolatility()
        self.currentPrice = self.mediumTermData.iloc[-1, 3]
        self.__setTechnicalIndicators()
        self.__createPredictionDataFrames()
        #self.__regressionPredictFuturePriceChanges(periods)


    def setSymbol(self, tickerSymbol):
        self.symbol = tickerSymbol
        self.__setTicker()
        self.__setData()
        
    def __setTicker(self):
        self.ticker = yf.Ticker(self.symbol)

    def __setData(self):
        self.longTermData = self.ticker.history(period = "10y", interval = "1wk").dropna()
        self.mediumTermData = self.ticker.history(period = "2y", interval = "1d").dropna()
        self.shortTermData = self.ticker.history(period = "1mo",  interval = "1h").dropna()

        self.__setVolatility()
        self.__setTechnicalIndicators()
        self.__createPredictionDataFrames()
        self.currentPrice = self.mediumTermData.iloc[-1, 3]

    def __setVolatility(self):
        self.annualizedVolatility = self.mediumTermData["Close"].std()/2
        self.estimatedDailyVolatility = self.annualizedVolatility/16
        self.estimatedWeeklyVolatility = self.annualizedVolatility/math.sqrt(52)
        self.actualWeeklyVolatility = self.shortTermData["Close"].std()
        self.actualDailyVolatility = self.shortTermData.iloc[-8:, 3].std()

    def displayDailyDistribution(self):
        mu = self.currentPrice
        sigma = self.actualDailyVolatility
        x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
        plt.plot(x, stats.norm.pdf(x, mu, sigma))
        plt.show()

    def displayWeeklyDistribution(self):
        mu = self.currentPrice
        sigma = self.actualWeeklyVolatility
        x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
        plt.plot(x, stats.norm.pdf(x, mu, sigma))
        plt.show()

    def displayAnnualDistribution(self):
        mu = self.currentPrice
        sigma = self.annualizedVolatility
        x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
        plt.plot(x, stats.norm.pdf(x, mu, sigma))
        plt.show()

    def getSymbol(self):
        return self.symbol
    
    def getTicker(self):
        return self.ticker

    def getLongTermData(self):
        return self.longTermData

    def getMediumTermData(self):
        return self.mediumTermData

    def getShortTermData(self):
        return self.shortTermData

    def getAnnualizedVolatility(self):
        return self.annualizedVolatility

    def getEstimatedDailyVolatility(self):
        return self.estimatedDailyVolatility

    def getEstimatedWeeklyVolatility(self):
        return self.estimatedWeeklyVolatility

    def getActualDailyVolatility(self):
        return self.actualDailyVolatility
    
    def getActualWeeklyVolatility(self):
        return self.actualWeeklyVolatility

    def __calculateADX (self, period = 14):
        """
        Calculates the Average Directional Index (ADX) values for the equity

        Inputs:
            period (int) - period to calculate the indidcator on, standard value is 14
        Outputs:
            adx (Series) - a column containing adx for the corresponding date
            pdi (Series) - a column containing postive directional index for the corresponding entry
            ndi (Series) - a column containing negative directional index for the corresponding entry
            adx_indicator (Series) - a column containing the trading signal based on the technical indicators value
        """
        terms = {"long": self.longTermData, "medium" : self.mediumTermData, "short" : self.shortTermData}
        
        for x in terms:

            df = terms[x]

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
                
            if x == "long":
                self.longTermADXSeries = adx
                self.currentLongTermADX = adx[-1]
            elif x == "medium":
                self.mediumTermADXSeries = adx
                self.currentMediumTermADX = adx[-1]
            else:
                self.shortTermADXSeries = adx
                self.currentShortTermADX = adx[-1]
    

    def __calculateAroon(self, period = 25):
        """
        Calculates the Aroon Indicator values for the equity

        Inputs:
            period (int) - period to calculate the indidcator on, standard value is 25
            
        Outputs:
            aroonup (Series) - a column containing Aroon Up values for the corresponding entry
            aroondown (Series) - a column containing Aroon Down values for the corresponding entry
        """   
        terms = {"long": self.longTermData, "medium" : self.mediumTermData, "short" : self.shortTermData}
        
        for x in terms:

            df = terms[x]

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

            if x == "long":
                self.longTermAroonUpSeries = aroonup
                self.longTermAroonDownSeries = aroondown
                self.currentLongTermAroonUp = aroonup[-1]
                self.currentLongTermAroonDown = aroondown[-1]
            elif x == "medium":
                self.mediumTermAroonUpSeries = aroonup
                self.mediumTermAroonDownSeries = aroondown
                self.currentMediumTermAroonUp = aroonup[-1]
                self.currentMediumTermAroonDown = aroondown[-1]
            else:
                self.shortTermAroonUpSeries = aroonup
                self.shortTermAroonDownSeries = aroondown
                self.currentShortTermAroonUp = aroonup[-1]
                self.currentShortTermAroonDown = aroondown[-1]


    def __calculateCCI(self, period = 20):
        """
        Calculates the Commodity Channel Index values for the equity

        Inputs:
            period (int) - period to calculate the indidcator on, standard value is 20
            
        Outputs:
            cci (Series) - a column containing cci values for the corresponding entry

        """  
        terms = {"long": self.longTermData, "medium" : self.mediumTermData, "short" : self.shortTermData}
        
        for x in terms:

            df = terms[x] 

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

            if x == "long":
                self.longTermCCISeries = cci
                self.currentLongTermCCI = cci[-1]
            elif x == "medium":
                self.mediumTermCCISeries = cci
                self.currentMediumTermCCI = cci[-1]
            else:
                self.shortTermCCISeries = cci
                self.currentShortTermCCI = cci[-1]


    def __calculateDisparity(self, period = 14):
        """
        Calculates the Disparity Index values for the equity

        Inputs:
            period (int) - period to calculate the indidcator on, standard value is 14
            
        Outputs:
            disparity (Series) - a column containing disparity values for the corresponding entry
        """
        terms = {"long": self.longTermData, "medium" : self.mediumTermData, "short" : self.shortTermData}
        
        for x in terms:

            df = terms[x] 

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

            if x == "long":
                self.longTermDisparitySeries = disparity
                self.currentLongTermDisparity = disparity[-1]
            elif x == "medium":
                self.mediumTermDisparitySeries = disparity
                self.currentMediumTermDisparity = disparity[-1]
            else:
                self.shortTermDisparitySeries = disparity
                self.currentShortTermDisparity = disparity[-1]

    def __roc(self, price, n):
        """
        Calculates the Rate of Change (ROC) of a price over n periods
        """
        return (price.diff(n)/price.shift(n))*100

    def roc(self, price, n):
        """
        Calculates the Rate of Change (ROC) of a price over n periods
        """
        return (price.diff(n)/price.shift(n))*100

    def __calculateKST(self, signal = 9):
        """
        Calculates the Know Sure Thing (KST) values for the equity

        Inputs:
            signal (int) - period to calculate the signal of indidcator on, standard value is 9
            
        Outputs:
            kst (Series) - a column containing KST for the corresponding entry
            kstsignal (Series) - a column containing KST signal for the corresponding entry
        """
        terms = {"long": self.longTermData, "medium" : self.mediumTermData, "short" : self.shortTermData}
            
        for x in terms:

            df = terms[x] 

            close = df["Close"]

            r1 = self.__roc(close, 10).rolling(10).mean()
            r2 = self.__roc(close, 15).rolling(10).mean()
            r3 = self.__roc(close, 20).rolling(10).mean()
            r4 = self.__roc(close, 30).rolling(15).mean()
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
            if x == "long":
                self.longTermKSTSeries = kst
                self.currentLongTermKST = kst[-1]
            elif x == "medium":
                self.mediumTermKSTSeries = kst
                self.currentMediumTermKST = kst[-1]
            else: #short term dataframe is not long enough will give all NaNs
                self.shortTermKSTSeries = kst 
                self.currentShortTermKST = kst[-1]



    def __calculateMACD (self, long = 26, short = 12, lSignal = 9):
        """
        Calculates the moving average convergence divergence values for the equity

        Inputs:
            long (int) - the length of longer EMA (general metric = 26)
            short (int) - the length of the shofter EMA (general metric = 12)
            signal (int) - the timeframe to compute the signal for the MACD (general metric = 9)
        Outputs:
            macd (Series) - a column containing macd for the corresponding entry
            signal (Series) - a column containing the signal for the corresponding entry
        """
        terms = {"long": self.longTermData, "medium" : self.mediumTermData, "short" : self.shortTermData}
            
        for x in terms:

            df = terms[x] 

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
                    MACDcrossover.append("BullishCrossover")
                elif macd1 > signal1 and macd2 < signal2:
                    MACDcrossover.append("BearishCrossover")
                else:
                    MACDcrossover.append("Neutral")

            MACDcrossover = pd.Series(MACDcrossover, index = df.index)
            

            if x == "long":
                self.longTermMACDSeries = macd
                self.currentLongTermMACD = macd[-1]
                self.longTermMACDCrossover = MACDcrossover
                self.currentLongTermMACDCrossover = MACDcrossover[-1]
            elif x == "medium":
                self.mediumTermMACDSeries = macd
                self.currentMediumTermMACD = macd[-1]
                self.mediumTermMACDCrossover = MACDcrossover
                self.currentMediumTermMACDCrossover = MACDcrossover[-1]
            else:
                self.shortTermMACDSeries = macd
                self.currentShortTermMACD = macd[-1]
                self.shortTermMACDCrossover = MACDcrossover
                self.currentShortTermMACDCrossover = MACDcrossover[-1]
            

    def __calculateRVI(self, period = 10):
        """
        Calculates the Relative Vigor Index (RVI) values for the equity

        Inputs:
            period (int) - period to calculate the indidcator on, standard value is 10
            
        Outputs:
            rvi (Series) - a column containing RVI for the corresponding entry
            rvisignal (Series) - a column containing RVI signal for the corresponding entry
        """
        terms = {"long": self.longTermData, "medium" : self.mediumTermData, "short" : self.shortTermData}
            
        for x in terms:
            df = terms[x]

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

            if x == "long":
                self.longTermRVISeries = rvi
                self.currentLongTermRVI = rvi[-1]
            elif x == "medium":
                self.mediumTermRVISeries = rvi
                self.currentMediumTermRVI = rvi[-1]
            else:
                self.shortTermRVISeries = rvi
                self.currentShortTermRVI = rvi[-1]
            
    def __calculateStochasticOscillator (self, period = 14, signal = 3):
        """
        Calculates the Stochastic Oscillator values for the equity

        Inputs:
            period (int) - period to calculate the indidcator on, standard value is 14
            signal (int) - the timeframe to compute the signal for the Stochastic Oscillator,
            standard value is 3
            
        Outputs:
            so (Series) - a column containing Stochastic Oscillator for the corresponding entry
            sosignal (Series) - a column containing Stochastic Oscillator signal for the 
                corresponding entry
        """
        terms = {"long": self.longTermData, "medium" : self.mediumTermData, "short" : self.shortTermData}
            
        for x in terms:

            df = terms[x] 

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

            if x == "long":
                self.longTermStochasticOscillatorSeries = so
                self.currentLongTermStochasticOscillator = so[-1]
            elif x == "medium":
                self.mediumTermStochasticOscillatorSeries = so
                self.currentMediumTermStochasticOscillator = so[-1]
            else:
                self.shortTermStochasticOscillatorSeries = so
                self.currentShortTermStochasticOscillator = so[-1]

    def __calculateWilliamsR(self, period = 14):
        """
        Calculates the Williams %R values for the equity
        Inputs:
            period (int) - period to calculate the indidcator on, standard value is 14
        Outputs:
            williamsr (Series) - a column containing Williams %R indicator for the corresponding entry
        """
        terms = {"long": self.longTermData, "medium" : self.mediumTermData, "short" : self.shortTermData}
                    
        for x in terms:

            df = terms[x]
        
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

            if x == "long":
                self.longTermWilliamsRSeries = williamsr
                self.currentLongTermWilliamsR = williamsr[-1]
            elif x == "medium":
                self.mediumTermWilliamsRSeries = williamsr
                self.currentMediumTermWilliamsR = williamsr[-1]
            else:
                self.shortTermWilliamsRSeries = williamsr
                self.currentShortTermWilliamsR = williamsr[-1]

    def __calculateRSI(self, period = 14):
        """
        Calculates the Relative Strength Index (RSI) values for the equity

        Inputs:
            period (int) - period to calculate the indidcator on, standard value is 14
        Outputs:
            rsi (Series) - a column containing rsi for the corresponding entry
        """
        terms = {"long": self.longTermData, "medium" : self.mediumTermData, "short" : self.shortTermData}
                    
        for x in terms:

            df = terms[x]

            close = df["Close"]  
            diff = close.diff()
            gain = []
            loss = []
            for i in range(len(diff)):
                if diff[i] < 0:
                    gain.append(0)
                    loss.append(diff[i])
                else:
                    gain.append(diff[i])
                    loss.append(0)
            gain = pd.Series(gain)
            gain.index = close.index

            loss = pd.Series(loss)
            loss.index = close.index

            gainEMA = gain.ewm(span = period - 1, adjust = False).mean()
            lossEMA = abs(loss.ewm(span = period - 1, adjust = False).mean())
            rs = gainEMA/lossEMA
            rsi = 100 - (100 / (1 + rs))

            rsisignal = []
            for i in range(len(rsi)):
                if rsi[i] < 30:
                    rsisignal.append("Buy")
                elif rsi[i] > 70:
                    rsisignal.append("Sell")
                else:
                    rsisignal.append("Neutral")

            if x == "long":
                self.longTermRSISeries = rsi
                self.currentLongTermRSI = rsi[len(rsi)-1]
            elif x == "medium":
                self.mediumTermRSISeries = rsi
                self.currentMediumTermRSI = rsi[len(rsi)-1]
            else:
                self.shortTermRSISeries = rsi
                self.currentShortTermRSI = rsi[len(rsi)-1]

    def __calculateOBV(self):
        terms = {"long": self.longTermData, "medium" : self.mediumTermData, "short" : self.shortTermData}
                    
        for x in terms:

            df = terms[x]

            close = df["Close"]
            volume = df["Volume"]
            obv = []
            obv.append(volume[0])
            

            for i in range(0,len(volume)-1):
                if close[i+1] > close[i]:
                    obv.append(obv[i] + volume[i+1])
                elif close[i+1] == close[i]:
                    obv.append(obv[i])
                else:
                    obv.append(obv[i] - volume[i+1])

            obv = pd.Series(obv, index = close.index)

            if x == "long":
                self.longTermOBVSeries = obv
                self.currentLongTermOBV = obv[len(obv)-1]
            elif x == "medium":
                self.mediumTermOBVSeries = obv
                self.currentMediumTermOBV = obv[len(obv)-1]
            else:
                self.shortTermOBVSeries = obv
                self.currentShortTermOBV = obv[len(obv)-1]
    
    def __setTechnicalIndicators(self):
        self.__calculateADX()
        self.__calculateAroon()
        self.__calculateCCI()
        self.__calculateDisparity()
        self.__calculateMACD()
        self.__calculateKST()
        self.__calculateStochasticOscillator()
        self.__calculateWilliamsR() 
        #Note: Stochastic Oscillator and WilliamsR are calculated using same
        #underlying and are perfectly collinear
        self.__calculateRSI()
        self.__calculateRVI()
        self.__calculateOBV()
    
    def __setFuturePrice(self, df, n):
        """
        Calculates the equity price n days in the future
        Inputs:
            df (DataFrame) - dataframe that contains the data of the equity of interest
            n (int) - number of days in future
        Outputs: 
            fp (Series) - the future prices of the equity 
        """
        fp = df["Close"].shift(-n)
        return fp

    def setPercentChange(self, df, n):
        """
        Adds a column to the dataframe with percent change of the stock n days in the future
        Inputs:
            df (DataFrame) - dataframe that contains the data of the equity of interest
            n (int) - number of days in future
        Outputs: 
            data (DataFrame) - dataframe with added column
        """
        data = df.copy()
        fp = self.__setFuturePrice(data, n)
        pc = (fp - data["Close"] )/ data["Close"]
        return pc

    def __createPredictionDataFrames(self):
        
        terms = {"long": self.longTermData, "medium" : self.mediumTermData, "short" : self.shortTermData}
                        
        for x in terms:

            df = terms[x].copy()
            if x == "long":
                df["ADX"] = self.longTermADXSeries
                df["AroonUp"] = self.longTermAroonUpSeries
                df["AroonDown"] = self.longTermAroonDownSeries
                df["CCI"] = self.longTermCCISeries
                df["Disparity"] = self.longTermDisparitySeries
                df["KST"] = self.longTermKSTSeries
                df["MACD"] = self.longTermMACDSeries
                df["RVI"] = self.longTermRVISeries
                df["RSI"] = self.longTermRSISeries
                #df["OBV"] = self.longTermOBVSeries
                df["StochasticOscillator"] = self.longTermStochasticOscillatorSeries
                self.longTermPredictionDF = df.drop(['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits'], axis = 1)
            elif x == "medium":
                df["ADX"] = self.mediumTermADXSeries
                df["AroonUp"] = self.mediumTermAroonUpSeries
                df["AroonDown"] = self.mediumTermAroonDownSeries
                df["CCI"] = self.mediumTermCCISeries
                df["Disparity"] = self.mediumTermDisparitySeries
                df["KST"] = self.mediumTermKSTSeries
                df["MACD"] = self.mediumTermMACDSeries
                df["RVI"] = self.mediumTermRVISeries
                df["RSI"] = self.mediumTermRSISeries
                #df["OBV"] = self.mediumTermOBVSeries
                df["StochasticOscillator"] = self.mediumTermStochasticOscillatorSeries
                self.mediumTermPredictionDF = df.drop(['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits'], axis = 1)
            else:
                df["ADX"] = self.shortTermADXSeries
                df["AroonUp"] = self.shortTermAroonUpSeries
                df["AroonDown"] = self.shortTermAroonDownSeries
                df["CCI"] = self.shortTermCCISeries
                df["Disparity"] = self.shortTermDisparitySeries
                df["KST"] = self.shortTermKSTSeries
                df["MACD"] = self.shortTermMACDSeries
                df["RVI"] = self.shortTermRVISeries
                df["RSI"] = self.shortTermRSISeries
                #df["OBV"] = self.longTermOBVSeries
                df["StochasticOscillator"] = self.shortTermStochasticOscillatorSeries
                self.shortTermPredictionDF = df.drop(['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits'], axis = 1)

    def __fitModel(self, X, Y):

        #linear_model = lm.LinearRegression(fit_intercept=True)

        #linear_model.fit(X, Y)

        #fitted = linear_model.predict(X)
        return #linear_model, fitted

    def __regressionPredictFuturePriceChanges(self, periods):
        """
        Predicts the future price of the equity n periods into the future
        """ 
        terms = {"long": self.longTermData, "medium" : self.mediumTermData, "short" : self.shortTermData}
                        
        for x in terms:
            if x == "long":
                df1 = self.longTermPredictionDF
                df2 = self.longTermData
                X2 = self.longTermPredictionDF
            elif x == "medium":
                df1 = self.mediumTermPredictionDF
                df2 = self.mediumTermData 
                X2 = self.mediumTermPredictionDF
            else:
                df1 = self.shortTermPredictionDF
                df2 = self.shortTermData
                X2 = self.shortTermPredictionDF

            df1["PercentChange"] = self.setPercentChange(df2, periods)

            df1.dropna(inplace = True)
            X = df1.loc[:, df1.columns != "PercentChange"]
            Y = df1["PercentChange"]

            model, fitted = self.__fitModel(X,Y)

            X2 = X2.copy().loc[:, df1.columns != "PercentChange"]
            X2.dropna(inplace = True)
            
            fitted2 = model.predict(X2)
            fitted2 = pd.Series(data = fitted2, index = X2.index)

            if x == "long":
                self.longTermPredictedChanges = fitted2
                self.currentLongTermPredictionP = fitted2[-1]
                self.currentLongTermPredictionV = self.currentPrice*(1+self.currentLongTermPredictionP)
            elif x == "medium":
                self.mediumTermPredictedChanges = fitted2
                self.currentMediumTermPredictionP = fitted2[-1]
                self.currentMediumTermPredictionV = self.currentPrice*(1+self.currentMediumTermPredictionP)
            else:
                self.shortTermPredictedChanges = fitted2
                self.currentShortTermPredictionP = fitted2[-1]
                self.currentShortTermPredictionV = self.currentPrice*(1+self.currentShortTermPredictionP)

       

    def findPercentile(self, percentChange, term, periods):
        if term == "long":
            sigma = self.actualWeeklyVolatility*periods
            mu = self.currentPrice
            
            percentile = stats.norm(loc = mu, scale = sigma).cdf(mu*(1+percentChange))


        elif term == "medium":
            sigma = self.actualDailyVolatility*periods
            mu = self.currentPrice
            
            percentile = stats.norm(loc = mu, scale = sigma).cdf(mu*(1+percentChange))
        
        else:
            sigma = self.actualDailyVolatility/6.5*periods
            mu = self.currentPrice
            
            percentile = stats.norm(loc = mu, scale = sigma).cdf(mu*(1+percentChange))
    
        return percentile

    def RMSE(self, actual, predicted):
        return ((predicted - actual) ** 2).mean() ** 0.5

    def backtest(self, close, _open, signal):
        if len(close) != len(_open) and len(close) != len(signal):
            print("Series arent the same length")
            return

        longbought = []
        shortbought = []
        longsold = []
        shortsold = []
        longdate = []
        shortdate = []
        holdings = 0
        leveraged = 0
        
        for i in range(len(close)-1):
            o = _open[i+1]
            if signal[i] in ("Long", "Buy", "Bullish"):
                if holdings < 0:
                    start = len(shortsold)
                    end = len(shortbought)
                    for i in range(start, end):
                        shortsold.append(o)
                    holdings = 0
                else:
                    longbought.append(o)
                    longdate.append(close.index[i])
                    holdings += 1
                    leveraged += o
            elif signal[i] in ("Short", "Sell", "Bearish"):
                if holdings > 0:
                    start = len(longsold)
                    end = len(longbought)
                    for i in range(start, end):
                        longsold.append(o)
                    holdings = 0
                else:
                    shortbought.append(o)
                    shortdate.append(close.index[i])
                    holdings -= 1
                    leveraged += o
            if len(longbought) != 0 and len(longbought) > len(longsold) and close.index[i] - longdate[-1] > dt.timedelta(days=30):
                x= len(longbought) - len(longsold)
                while x < len(longbought) and 1.15*longbought[x] < close[i]:
                    longsold.append(close)
                    x+=1

            if len(shortbought) != 0 and len(shortbought) < len(shortsold) and close.index[i] - shortdate[-1] > dt.timedelta(days=30):
                x = len(shortbought) - len(shortsold)
                while x < len(shortbought) and shortbought[x] > 1.15*close[i]:
                    shortsold.append(close)
                    x+=1
                
        if holdings > 0:
            while len(longbought) > len(longsold):
                longsold.append(close[-1])
        elif holdings < 0:
            while len(shortbought) > len(shortsold):
                shortsold.append(close[-1])
        if len(longdate) > 0:
            longprofit = []
            longpercentreturn = []

            for i in range(len(longbought)):
                longprofit.append(longsold[i]-longbought[i])
                longpercentreturn.append(longprofit[i]/longbought[i])

            longdf = pd.DataFrame(index= longdate)

            indicator_col = [1]*len(longsold)
            longdf["Long"] = pd.Series(indicator_col, index = longdate)
            longdf["Bought"] = pd.Series(longbought, index = longdate)
            longdf["Sold"] = pd.Series(longsold, index = longdate)
            longdf["Return"] = pd.Series(longprofit, index = longdate)
            longdf["PercentReturn"] = pd.Series(longpercentreturn, index = longdate)
            results = longdf

        if len(shortdate) > 0:
            shortprofit = []
            shortpercentreturn = []
            for i in range(len(shortbought)):
                shortprofit.append(shortbought[i]-shortsold[i])
                shortpercentreturn.append(shortprofit[i]/shortsold[i])

            
            shortdf = pd.DataFrame(index= shortdate)

            indicator_col = [0]*len(shortdate)
            shortdf["Long"] = pd.Series(indicator_col, index = shortdate)
            shortdf["Bought"] = pd.Series(shortbought, index = shortdate)
            shortdf["Sold"] = pd.Series(shortsold, index = shortdate)
            shortdf["Return"] = pd.Series(shortprofit, index = shortdate)
            shortdf["PercentReturn"] = pd.Series(shortpercentreturn, index = shortdate)
            results = shortdf

        if len(shortdate) > 0 and len(longdate) > 0:
            results = pd.concat([longdf, shortdf])
        
        if len(shortbought) > 0 or len(longdate) > 0:
            n_trades = len(shortbought) + len(longbought)
            total_percent_return = results["Return"].sum()/leveraged
            total_return = results["Return"].sum()
            startdate = close.index[0]
            enddate = close.index[-1]

            print("""---------------------------------------------------------------
            Backtest Results
            ---------------------------------------------------------------
            From: {} To: {}
            ---------------------------------------------------------------
                        Number of trades: {}
            ---------------------------------------------------------------
            Profit: ${:.2f}	Leveraged: ${:.2f}

                % Return: {:.2f}%
            ---------------------------------------------------------------
                        |Minimum|Median|Average|Maximum|
            ---------------------------------------------------------------
            Return         |{:^7.2f}|{:^6.2f}|{:^7.2f}|{:^7.2f}|  
            ---------------------------------------------------------------
            Percent Return |{:^7.2f}|{:^6.2f}|{:^7.2f}|{:^7.2f}|
            _______________________________________________________________""".format(
            startdate, enddate, n_trades, total_return, leveraged, total_percent_return*100,
            results["Return"].min(), results["Return"].median(), results["Return"].mean(), 
            results["Return"].max(), results["PercentReturn"].min()*100, results["PercentReturn"].median()*100,
            results["PercentReturn"].mean()*100, results["PercentReturn"].max()*100))
            return results
        

    def calculateOBVVelocityIndicator(self, OBVSeries, threshold = 0.85):
        x = list(range(0,len(OBVSeries)))
        coefs = np.polyfit(x, OBVSeries, 20)
        lofb = np.poly1d(coefs)
        OBV_deriv = lofb.deriv()
        OBV_velocity = []
        for i in x:
            OBV_velocity.append(OBV_deriv(i))

        mu = np.mean(OBV_velocity)
        sigma = np.std(OBV_velocity)

        percentile = stats.norm(loc=mu, scale = sigma).cdf(OBV_velocity)

        indicator = []
        
        for i in range(len(percentile)):
            if percentile[i] > threshold:
                indicator.append("StrongTrend")
            else:
                indicator.append("Inconclusive")

        return pd.Series(indicator, index = OBVSeries.index)




    def calculateOBV(self, close, volume):
        obv = []
        obv.append(volume[0])
        
        for i in range(0,len(volume)-1):
            if close[i+1] > close[i]:
                obv.append(obv[i] + volume[i+1])
            elif close[i+1] == close[i]:
                obv.append(obv[i])
            else:
                obv.append(obv[i] - volume[i+1])

        obv = pd.Series(obv, index = volume.index)
        return obv



    def calculateAroon(self, high, low, period=25):
    
        aroonup = 100 * high.rolling(period + 1).apply(lambda x: x.argmax()) / period
        aroondown = 100 * low.rolling(period + 1).apply(lambda x: x.argmin()) / period

        return pd.Series(aroonup, index = high.index), pd.Series(aroondown, index = high.index)