import numpy as np
import pandas as pd
import networkx as nx
import scipy.stats as stats
import statsmodels.tsa.stattools as stattools
import statsmodels.tsa.statespace.tools as tools

import statistics.time_series as ts
import statistics.model as mod
import statistics.data_analysis as da
import statistics.random_variable as ranv
import finance.technical as tech
import finance.iex as iex

connection = iex.Connection()

# all time series used are logarithms of the prices
class Screener:
    """basically just a list of symbols in the same industry:
    takes a list of tickers... uses log prices"""
    def __init__(self, tickers, length=100):
        assert isinstance(tickers, list)
        self.tickers = tickers
        self.length = length
        self.size = len(self.tickers)
        self.encoding_dict = {i: self.tickers[i] for i in range(self.size)}
        self.series_dict = self.get_series_dict()

    def get_series_dict(self):
        series_dict = dict()

        for ticker in self.tickers:
            ticker_series = connection.get_series(ticker, self.length)
            ticker_series = np.log(ticker_series)
            series_dict[ticker] = ticker_series

        return series_dict

    def coint_network(self):
        graph = nx.Graph()
        for i in range(self.size):
            for j in range(self.size):
                if i < j:
                    seri = self.encoding_dict[i].values
                    serj = self.encoding_dict[j].values
                    if stattools.coint(seri, serj)[1]<.01:
                        graph.add_edge(i,j)

        return graph

    def trend_dict(self, alpha):
        """returns the tickers that have a statistically significant
         trend of the logprices (using ARIMA(1,1,0) model)"""

        trend_dict = dict()
        for ticker in self.tickers:
            ser = self.series_dict[ticker].values
            diff = tools.diff(ser)
            if stattools.adfuller(ser)[1] > .01:
                continue

            mu, sig, n = np.mean(diff), np.std(diff, ddof=1), len(diff)
            intv = stats.norm.interval(1-alpha, loc=0, scale=sig/np.sqrt(n)

            if mu < intv[0] or mu > intv[1]:
                trend_dict[ticker] = intv

        return trend_dict








    def geometric_dict(self, trend_dict, horizon=20):
        geo_dict = dict()
        bin_num = np.int(np.floor(self.series_length/2.3))
        for ticker in trend_dict.keys():
            if trend_dict[ticker] > 0:
                differenced = ts.diff(self.series_dict[ticker])

                #for distributions
                args = [differenced.values, bin_num]
                best_fit = da.fit_data(*args)
                dist = best_fit[0]
                params = best_fit[1]
                dist_args = params[:-2]
                loc = params[-2]
                scale = params[-1]

                #for geomeric motion
                avg = np.average(differenced.values)
                sigma = np.std(differenced.values, ddof=1)
                mu = avg + sigma**2

                class rv(stats.rv_continuous):
                    def _pdf(self,x):
                        return ranv.normalize_pdf(dist.pdf(x, *dist_args, loc=loc, scale=scale))

                function = mod.geometric_motion
                args = [rv, horizon, self.series_dict[ticker].values[0], mu, sigma]
                predicted_vals = mod.monte_carlo(function, 1000, *args)
                geo_dict[ticker] = {'avg': np.average(predicted_vals), 'std': np.std(predicted_vals, ddof=1)}

        return geo_dict

    def extreme_val_dict(self, trend_dict, moving_length, threshold):
        ext_vals = dict()
        for equity in trend_dict.keys():
            if trend_dict[equity] != 0:
                series = self.series_dict[equity]
                args = [series, np.std, moving_length, 2, True]
                [low, high] = tech.bands(*args)

                if series[-1] <= low + np.multiply(threshold, series[-1]):
                    ext_vals[equity] = 'low'
                elif series[-1] >= high - np.multiply(threshold, series[-1]):
                    ext_vals[equity] = 'high'

        return ext_vals
