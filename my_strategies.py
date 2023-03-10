import pandas as pd
from my_functions import get_from_db
from backtesting import Strategy 
from backtesting.lib import crossover, cross 

def get_data_for_bt(symbol, model, con):

    query = "SELECT Date,open,high,low,close,volume FROM History WHERE symbol='{}'".format(symbol)

    data = pd.read_sql(query, con, index_col="Date", parse_dates={"Date": "%Y%m%d %H:%M:%S"}
                    ).sort_index().rename(columns={'open': 'Open',
                                                        'high': 'High',
                                                        'low': 'Low',
                                                        'close': 'Close',
                                                        'volume': 'Volume'})
    data.index.name = symbol, model
    
    return data


def get_mid_term_expected_price(symbol, model, first_week, last_week, con):

    forecast = get_from_db('Forecast', symbol, con, model_type=model)
    forecast = forecast[(forecast.relevance!=0)]
    return forecast.price.rolling(last_week - first_week).mean().shift(-last_week)


def get_next_week_expected_price(symbol, model, con):

    forecast = get_from_db('Forecast', symbol, con, model_type=model)
    forecast = forecast[(forecast.relevance!=0)]
    return forecast.price.shift(-1)


def resample_expected_price(data, expected_price):

    daily_expected_price = expected_price.resample('D').ffill()
    
    return daily_expected_price.reindex(data.index)


class MidTermEstimation(Strategy):

    #params changes in bt.run
    value_factor = 0.2
    usable_weeks = (13, 26)
    con = ''
    

    def init(self):

        symbol = self.data.index.name[0]
        model= self.data.index.name[1]

        mid_term_expected_price = get_mid_term_expected_price(symbol, model, self.usable_weeks[0], self.usable_weeks[1], self.con)

        self.future_price = self.I(resample_expected_price, self.data, mid_term_expected_price, name='Mid-term expected price')

        #next_week_expected_price = get_next_week_expected_price(symbol, model)

        #self.week_future_price = self.I(resample_expected_price, data, next_week_expected_price, name='week_future_price')

    def next(self):

        if self.position:

            if cross(self.future_price, self.data.Close):

                self.position.close()

        else:

            if crossover(self.future_price, self.data.Close * (1 + self.value_factor)):
                self.buy() #size=50

            if crossover(self.data.Close * (1 - self.value_factor), self.future_price):
                self.sell()