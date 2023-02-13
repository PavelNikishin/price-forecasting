# my_functions

import sys
import json
import yfinance as yf
import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy.stats as scs
from prophet import Prophet
from datetime import datetime


class YahooTicker:
    def __init__(self, symbol):

        _info_ = yf.Ticker(symbol).info

        self.info = {
            "symbol": symbol,
            # "name": ticker_names[symbol],
            "underlyingSymbol": _info_.get("underlyingSymbol"),
            "shortName": _info_.get("shortName"),
            "exchange": _info_.get("exchange"),  # yf.Ticker(symbol).fast_info['exchange'],
            "quoteType": _info_.get("quoteType"),
        }

    def add_history(
        self,
        start=None,
        end=None,
        interval="1d",
        progress=False,
        period="max",
        rounding=False,
        back_adjust=True,  # dont use yahoo Adj Close
        continuous_method=None,  # how make the continuous futures
    ):

        self.history = yf.download(
            self.info["symbol"],
            start=start,
            end=end,
            interval=interval,
            progress=progress,
            period=period,
            rounding=rounding,
            back_adjust=back_adjust,
        )

        """
        if continuous_method:  # continuous method  can only be used for gluing futures prices
            if self.info.get('quoteType') == 'FUTURE':
                pass
                # in the future
                # make_continuous_futures(data, method=continuous_method)

            else:
                print('Continuous method can only be used for gluing futures prices')
                """


class All_residuals():

    '''
    Used in get_residuals_fig for streamlit
    '''

    def __init__(self, ys, metrics_list, model_fnames):

        metric_df = pd.DataFrame({fname: pd.Series(dtype='float') for fname in model_fnames})

        for metric in metrics_list:

            df = pd.DataFrame()

            for model_fname in model_fnames:

                if metric == 'absolute_error':
                    df[model_fname] = abs(ys.close - ys[model_fname])
                    metric_df.at['Средняя абсолютная ошибка (MAE)', model_fname] = df[model_fname].mean()

                elif metric == 'absolute_percentage_error':
                    df[model_fname] = abs(ys.close - ys[model_fname]) / abs(ys.close) * 100
                    metric_df.at['Средняя абсолютная процентная ошибка (MAPE)', model_fname] = df[model_fname].mean()

                elif metric == 'root_mean_square_error':
                    df[model_fname] = pow(ys.close - ys[model_fname], 2)  # without root
                    metric_df.at['Корень из среднеквадратичной ошибки (RMSE)', model_fname] = np.sqrt(df[model_fname].mean())

                elif metric == 'max_error':
                    metric_df.at['Максимальная ошибка', model_fname] = (abs(ys.close - ys[model_fname])).max()

            setattr(self, metric, df)

        self.metric_df = metric_df


def get_model_params(model_name, symbol):
    with open("./files/model_parameters.json", "r") as read_file:
        json_data = json.load(read_file)
        return json_data[symbol][model_name]


def get_ticker_data(symbol, con, start=None, end=None):

    # data = ticker.history['Close'].to_frame(name='close')

    data = get_from_db("History", symbol, con, start=start, end=end)

    resample_data = data.close.resample("W-FRI")

    weekly_data = resample_data.mean().fillna(method="ffill").to_frame(name="close")

    # drop incomplete week
    if resample_data.count()[-1] < 5:

        weekly_data.drop(weekly_data.tail(1).index, inplace=True)

    return data, weekly_data


def get_symbol_list(param_file_path="./files/model_parameters.json"):

    with open(param_file_path, "r") as read_file:
        json_data = json.load(read_file)
        return list(json_data.keys())


# Define db funcs

def get_from_db(table, symbol, con, model_type=None, start=None, end=None):

    if table == "History":

        select_columns = "Date, close"

    elif table == "Forecast":

        select_columns = "*"

    else:
        sys.exit("No such table")

    if start is None and end is None:

        dates = ""

    elif start is None:

        dates = "AND Date BETWEEN \"1900-01-01 00:00:00\" AND '{} 00:00:00'".format(
            end)

    elif end is None:

        dates = "AND Date BETWEEN '{} 00:00:00' AND \"3999-12-31 00:00:00\"".format(
            start
        )

    else:

        dates = "AND Date BETWEEN '{} 00:00:00' AND '{} 00:00:00'".format(
            start, end)

    if table == "Forecast" and model_type is not None:

        model = "AND model='{}'".format(model_type)

    else:

        model = ""

    query = "SELECT {} FROM {} WHERE symbol='{}' {} {}".format(
        select_columns, table, symbol, dates, model
    )

    return pd.read_sql(
        query, con, index_col="Date", parse_dates={"Date": "%Y%m%d %H:%M:%S"}
    ).sort_index()


def relevance_update(symbol, model_type, con, freq, start="1900-01-01", end="3999-12-31"):
    """
    set the relevance of the forecast:
        0 - there is a more recent forecast or forecast made retroactively
        1 - forecast is relevant
        2 - model fitted values
    """

    data = get_from_db("Forecast", symbol, con, model_type=model_type, start=start, end=end)

    # determine relevance
    data = data[data.relevance != 2]

    into_the_future = data[data.index >= data.recorded]
    into_the_past = data[data.index < data.recorded]

    for date, group in into_the_future.groupby("Date"):
        if len(group) > 1:
            into_the_future['relevance'][date] = np.where(
                group['recorded'] == group['recorded'].max(), 1, 0)
        else:
            into_the_future['relevance'][date] = 1

    into_the_past['relevance'] = 0

    # connecting the data back
    new_data = pd.concat([into_the_past, into_the_future], axis=0)
    new_data.sort_values(['Date', 'recorded'], inplace=True)

    to_sql = new_data[['recorded', 'relevance']].reset_index(level=0)
    to_sql['Date'] = to_sql['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')
    to_sql = to_sql[['relevance', 'Date', 'recorded']].values.tolist()

    # add filtering
    for r in to_sql:
        r.extend([symbol, model_type, freq])

    query = """
        UPDATE Forecast SET relevance = ? WHERE Date = ? AND recorded = ?
        AND symbol = ? AND model = ? AND freq = ?

    """
    with con:
        con.executemany(query, to_sql)


def add_to_db(table, symbol, add_data, con, freq="W-FRI", model_type=None):
    """
    add new values only if date does not exist
    """

    if table == "History":

        existing_data = get_from_db(
            table,
            symbol,
            con,
            model_type=model_type,
            start=add_data.index.min().strftime("%Y-%m-%d"),
            end=add_data.index.max().strftime("%Y-%m-%d"),
        )
        existing_dates = existing_data.index.strftime("%Y-%m-%d").values

        new_data = add_data.drop(existing_dates, axis=0)

        if not new_data.empty:

            new_data.assign(symbol=symbol).to_sql(
                table, con=con, if_exists="append")

    if table == "Forecast":

        if not model_type:

            sys.exit("specify the model type")

        today = datetime.now().strftime("%Y-%m-%d")

        existing_data = get_from_db(
            table,
            symbol,
            con,
            model_type=model_type,
            start=add_data.index.min().strftime("%Y-%m-%d"),
            end=add_data.index.max().strftime("%Y-%m-%d"),
        )

        existing_data = existing_data[existing_data.recorded >= today]

        existing_dates = existing_data.index.strftime("%Y-%m-%d").values

        new_data = add_data.drop(existing_dates, axis=0)

        if not new_data.empty:

            new_data.rename(columns={"forecast": "price"}).assign(
                symbol=symbol,
                model=model_type,
                freq=freq,
                recorded=today,
            ).to_sql(table, con=con, if_exists="append")

            relevance_update(symbol, model_type, con, freq=freq,
                             start=new_data.index.min().strftime("%Y-%m-%d"),
                             end=new_data.index.max().strftime("%Y-%m-%d"))


def make_arima_model(symbol, data=None, weekly_data=None):

    # use parameters
    model_params = get_model_params('arima', symbol)
    temp_model_params = {
        (k, v) for k, v in model_params.items() if k in ('week', 'boxcox')}

    if model_params['week']:

        ts = weekly_data

    else:

        ts = data

    if model_params['boxcox']:

        ts['close_bx'], lmbd = scs.boxcox(ts.close)
        temp_model_params['lmbd'] = lmbd
        ts = ts['close_bx']

    order, seasonal_order = model_params['order'], model_params['seasonal_order']

    p, d, q, P,  = order['p'], order['d'], order['q'], seasonal_order['P']
    D, Q, s = seasonal_order['D'], seasonal_order['Q'], seasonal_order['s']

    # fit model
    model = sm.tsa.statespace.SARIMAX(ts, order=(
        p, d, q), seasonal_order=(P, D, Q, s)).fit(disp=-1)

    fittedvalues = pd.concat([model.get_prediction().predicted_mean,
                              model.get_prediction().conf_int(alpha=0.05)], axis=1)

    # making a shift on s+d steps, because these values were unobserved by the model
    fittedvalues[:s+d] = np.NaN

    fittedvalues.rename(columns={'predicted_mean': 'fittedvalues', 'lower close': 'conf_int_lower',
                                 'upper close': 'conf_int_upper'}, inplace=True)

    if model_params['boxcox']:
        pass  # fittedvalues

    return model, fittedvalues, temp_model_params


def make_prophet_model(symbol, data=None, weekly_data=None):
    '''
    uses weekly data if it is provided
    '''

    # use parameters

    model_params = get_model_params('prophet', symbol)

    # prepare data

    if weekly_data is not None:

        df = weekly_data.close.reset_index()
        temp_model_params = {'week': True}

    else:

        df = data.close.reset_index()

    df.columns = ['ds', 'y']

    # fit model
    Prophet_model = Prophet(interval_width=0.95, daily_seasonality=False, **model_params)
    Prophet_model.fit(df)

    forecast = Prophet_model.predict()
    forecast.set_index('ds', inplace=True)

    fittedvalues = forecast[['yhat', 'yhat_lower', 'yhat_upper']].rename_axis('Date')

    fittedvalues.rename(columns={'yhat': 'fittedvalues', 'yhat_lower': 'conf_int_lower',
                                 'yhat_upper': 'conf_int_upper'}, inplace=True)

    return Prophet_model, fittedvalues, temp_model_params


def make_model(model_type, symbol, data=None, weekly_data=None):

    if model_type == 'arima':

        return make_arima_model(symbol, data=data, weekly_data=weekly_data)

    if model_type == 'prophet':

        return make_prophet_model(symbol, data=data, weekly_data=weekly_data)


def get_forecast(model, n_steps, temp_model_params=None):

    if model.__class__.__module__ == 'statsmodels.tsa.statespace.sarimax':

        #
        # keep in mind boxcox
        #

        forecast = pd.DataFrame(model.forecast(
            steps=n_steps).rename("forecast"))
        forecast.index.names = ['Date']
        forecast[['conf_int_lower', 'conf_int_upper']] = model.get_forecast(
            steps=n_steps).conf_int(alpha=0.05)

    if model.__class__.__module__ == 'prophet.forecaster':

        if temp_model_params is not None and temp_model_params['week']:
            freq = 'W-FRI'
        else:
            freq = 'D'

        future = model.make_future_dataframe(
            periods=n_steps, freq=freq, include_history=False)

        if freq == 'D':

            # drop weekends
            future = future[future['ds'].dt.dayofweek.between(0, 4)]

        forecast = model.predict(future)
        forecast.set_index('ds', inplace=True)

        forecast = forecast[['yhat', 'yhat_lower', 'yhat_upper']].rename(
            columns={'yhat': 'forecast', 'yhat_lower': 'conf_int_lower',
                     'yhat_upper': 'conf_int_upper'}
        )
        forecast.index.names = ['Date']

    return forecast


def get_table_from_db(table, con):

    if table not in ["History", "Forecast"]:

        sys.exit("No such table")

    query = "SELECT * FROM {}".format(table)

    return pd.read_sql(
        query, con, index_col="Date", parse_dates={"Date": "%Y%m%d %H:%M:%S"}
    ).sort_index()


def add_fittedvalues_to_db(symbol, add_data, con, model_type, freq="W"):
    '''
    use onle once!
    '''

    today = datetime.now().strftime("%Y-%m-%d")

    existing_data = get_from_db(
        'Forecast',
        symbol,
        con,
        model_type=model_type,
        start=add_data.index.min().strftime("%Y-%m-%d"),
        end=add_data.index.max().strftime("%Y-%m-%d"),
    )

    existing_data = existing_data[existing_data.recorded >= today]

    existing_dates = existing_data.index.strftime("%Y-%m-%d").values

    new_data = add_data.drop(existing_dates, axis=0)

    if not new_data.empty:

        new_data.rename(columns={"fittedvalues": "price"}).assign(symbol=symbol, model=model_type,
                                                                  freq=freq, recorded=today, relevance=2
                                                                  ).to_sql('Forecast', con=con, if_exists="append")


def update_history(con, symbol_list=None, start='2012-01-01', end=None):

    if not symbol_list:

        symbol_list = get_symbol_list()

    for symbol in symbol_list:
        ticker = YahooTicker(symbol)
        ticker.add_history(start=start, end=end)
        add_to_db('History', symbol, ticker.history, con)

    max_history_date = pd.read_sql('SELECT MAX(Date) FROM History', con, parse_dates={"MAX(Date)": "%Y%m%d %H:%M:%S"}
                                   ).loc[0][0].strftime("%Y-%m-%d")

    with open("./files/flags.json", "r") as read_file:
        json_data = json.load(read_file)

    json_data['last_history_update'] = datetime.now().strftime("%Y-%m-%d")

    with open("./files/flags.json", "w") as json_file:
        json.dump(json_data, json_file)

    with open("./files/jobs_log.txt", "a") as f:
        f.write(str(
            datetime.now().strftime("%Y-%m-%d %H:%M")
            + "  update_history executed for "
            + str(symbol_list)
            + ' by ' + max_history_date
        ))
        f.write("\n")


def weekly_forecasting(con, symbol_list=None, add_fittedvalues=False):

    if not symbol_list:

        symbol_list = get_symbol_list()

    model_names = pd.read_sql('SELECT * FROM Model', con).short_name.to_list()

    for symbol in symbol_list:

        data, weekly_data = get_ticker_data(symbol, con)

        for model_name in model_names:

            model, fittedvalues, temp_model_params = make_model(model_name, symbol, data, weekly_data)

            forecast = get_forecast(model, 52, temp_model_params)

            add_to_db('Forecast', symbol, forecast, con, model_type=model_name)

            if add_fittedvalues:

                add_fittedvalues_to_db(symbol, fittedvalues, con, model_name, freq="W-FRI")

    next_forecasting = pd.read_sql(
        'SELECT MIN(Date) FROM Forecast WHERE relevance = 1 AND recorded = (SELECT MAX(recorded) FROM Forecast)',
        con, parse_dates={"MIN(Date)": "%Y-%m-%d"}
    ).loc[0][0].strftime("%Y-%m-%d")

    with open("./files/flags.json", "r") as read_file:
        json_data = json.load(read_file)

    json_data['next_forecasting'] = next_forecasting

    with open("./files/flags.json", "w") as json_file:
        json.dump(json_data, json_file)

    with open("./files/jobs_log.txt", "a") as f:
        f.write(str(
            datetime.now().strftime("%Y-%m-%d %H:%M")
            + "  weekly_forecasting executed for "
            + str(symbol_list)
            + ' next_forecasting is ' + next_forecasting
        ))
        f.write("\n")


def force_data_update(con):

    update_history(con)

    weekly_forecasting(con)
