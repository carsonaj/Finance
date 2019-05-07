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
import market.iex as iex
import matplotlib.pyplot as plt

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
        self.series_dict = self.series_dict()

    def series_dict(self):
        series_dict = dict()

        for ticker in self.tickers:
            ticker_series = connection.get_series(ticker, self.length)
            ticker_series = np.log(ticker_series)
            series_dict[ticker] = ticker_series

        return series_dict

    def coint_graph(self, alpha):
        n = self.size
        graph = nx.Graph()
        for i in range(n):
            graph.add_node(i, pos=(np.cos(i), np.sin(i)))
        for i in range(n):
            for j in range(n):
                if i < j:
                    seri = self.series_dict[self.encoding_dict[i]].values
                    serj = self.series_dict[self.encoding_dict[j]].values
                    p = stattools.coint(seri, serj)[1]
                    if p <= alpha:
                        graph.add_edge(i,j, weight=p)

        return graph

    def trend_dict(self, alpha1=.01, alpha2=.01):
        """returns the tickers that have a significant
         trend(drift) of the logprices (using ARIMA(1,1,0) model)

         1-alpha1 gives rejection region of H0:no trend
         1-alph2 gives confidence interval of trend """

        trend_dict = dict()
        for ticker in self.tickers:
            ser = self.series_dict[ticker].values
            diff = tools.diff(ser)
            if stattools.adfuller(diff)[1] > .01:
                continue

            mu, sig, n = np.mean(diff), np.std(diff, ddof=1), len(diff)
            dist0 = stats.norm(loc=0, scale=sig/n)
            rrc = dist0.interval(1-alpha1)
            if mu < rrc[0] or mu > rrc[1]:
                dist1 = stats.norm(loc=mu, scale=sig/n)
                conf_int = dist1.interval(1-alpha2)
                trend_dict[ticker] = conf_int

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
