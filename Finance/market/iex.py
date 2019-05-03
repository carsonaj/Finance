import requests
import json
import numpy as np
import pandas as pd

class Connection:
    def __init__(self):
        self._api_url = 'https://api.iextrading.com/1.0'

    @property
    def api_url(self):
        return self._api_url

    #length up to 250
    def get_series(self, ticker, length):
        endpoint = '/stock/'+ticker+'/chart/1y'
        parameters = {'chartLast': length}
        response = requests.get(self.api_url+endpoint, params=parameters)
        json_data = response.json()

        dates = [entry['date'] for entry in json_data]
        date_index = pd.DatetimeIndex(dates)
        close = [entry['close'] for entry in json_data]
        prices = np.array(close, dtype='float64')

        series = pd.Series(data=prices, index=date_index)
        return series
