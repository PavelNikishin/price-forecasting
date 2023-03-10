
import sqlite3
import numpy as np
import pandas as pd
import json
from datetime import datetime, date, timedelta
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from backtesting import Backtest

from my_functions import All_residuals, get_symbol_list, get_from_db, get_ticker_data, update_history, weekly_forecasting
from my_strategies import MidTermEstimation, get_data_for_bt

st.set_page_config(layout="wide")

con = sqlite3.connect('./files/database.db')


@st.experimental_memo(ttl=86400)  # 24 hours
def get_forecast_fig(symbol, model, freq='W', con=con):

    # data
    if freq == 'W':
        _, selected_history = get_ticker_data(symbol, con)

    selected_forecast = get_from_db('Forecast', symbol, con, model_type=model)

    selected_history['relevance'] = 1

    # filtering
    fittedvalues = selected_forecast[(selected_forecast.relevance == 2) & (selected_forecast.price.notna())]
    relevance_forecast = selected_forecast[(selected_forecast.relevance == 1)]

    fig_h = go.Figure()
    fig_h.add_trace(go.Scatter(x=selected_history.index, y=selected_history["close"],
                               name="closing price",
                               line_width=1.5,
                               mode="lines", line_color='rgba(251, 13, 13, 0.6)'
                               ))

    fig_fv = go.Figure()
    fig_fv.add_trace(go.Scatter(x=fittedvalues.index, y=fittedvalues['conf_int_lower'],
                                mode='lines',
                                line_width=0.1,
                                legendgroup=1,
                                showlegend=False,
                                name='conf_int_lower',
                                line_color=px.colors.qualitative.Pastel[5]))
    fig_fv.add_trace(go.Scatter(
        x=fittedvalues.index, y=fittedvalues['conf_int_upper'],
        fill='tonexty',
        opacity=0.5,
        legendgroup=1,
        name='In-sample 95% confidence interval',
        mode='none',
        fillcolor=px.colors.qualitative.Pastel1[1]))

    fig_fv.add_trace(go.Scatter(
        x=fittedvalues.index, y=fittedvalues['conf_int_upper'],
        showlegend=False,
        legendgroup=1,
        line_width=0.1,
        name='conf_int_upper',
        mode='lines', line_color=px.colors.qualitative.Dark2[2]))

    fig_fv.add_trace(go.Scatter(
        x=fittedvalues.index, y=fittedvalues['price'],
        line_width=1.5,
        name='In-sample prediction',
        mode='lines', line_color=px.colors.qualitative.Light24[2]))

    fig_rf = go.Figure()

    fig_rf.add_trace(go.Scatter(x=relevance_forecast.index, y=relevance_forecast['conf_int_lower'],
                                mode='lines',
                                line_width=0.1,
                                legendgroup=2,
                                showlegend=False,
                                name='conf_int_lower',
                                line_color=px.colors.qualitative.Pastel[5]))

    fig_rf.add_trace(go.Scatter(
        x=relevance_forecast.index, y=relevance_forecast['conf_int_upper'],
        fill='tonexty',
        opacity=0.5,
        legendgroup=2,
        name='Out-of-sample 95% confidence interval',
        mode='none',
        fillcolor=px.colors.qualitative.Pastel[5]))

    fig_rf.add_trace(go.Scatter(
        x=relevance_forecast.index, y=relevance_forecast['conf_int_upper'],
        showlegend=False,
        legendgroup=2,
        line_width=0.1,
        name='conf_int_upper',
        mode='lines', line_color=px.colors.qualitative.Dark2[2]))

    fig_rf.add_trace(go.Scatter(
        x=relevance_forecast.index, y=relevance_forecast['price'],
        line_width=1.5,
        name='Out-of-sample prediction',
        mode='lines', line_color=px.colors.qualitative.Plotly[0]))

    full_forecast_fig = go.Figure(data=fig_fv.data + fig_rf.data + fig_h.data, layout=fig_h.layout)
    # full_forecast_fig.update_layout(legend_title_text='Trend')

    # Add range slider
    full_forecast_fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=2,
                         label="1+1 y",
                         step="year",
                         stepmode="backward"),
                    dict(count=6,
                         label="5+1 y",
                         step="year",
                         stepmode="backward"),
                    dict(count=11,
                         label="10+1 y",
                         step="year",
                         stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(
                visible=True,
            ),
            type="date"
        ),
        height=750
    )

    # Translate labels
    ru_names = {'closing price': 'Цена закрытия',
                'In-sample prediction': 'In-sample прогноз',
                'Out-of-sample prediction': 'Прогноз',
                'conf_int_upper': 'conf_int_upper',
                'conf_int_lower': 'conf_int_lower',
                'In-sample 95% confidence interval': 'in-sample 95% доверительный интервал',
                'Out-of-sample 95% confidence interval': '95% доверительный интервал'
                }

    full_forecast_fig.for_each_trace(lambda t: t.update(name=ru_names[t.name]))

    full_forecast_fig.update_layout(
        title="Прогноз цены",
        yaxis_title=None,
        font=dict(
            family="Source Sans Pro",
            size=16
        ),
        legend=dict(
            traceorder='reversed',
            orientation="h",
            itemwidth=45,
            yanchor="bottom",
            y=1.01,
            xanchor="right",
            x=1
        )

    )

    return full_forecast_fig


@st.experimental_memo(ttl=86400)  # 24 hours
def get_residuals_fig(symbol, model_fnames, models, metric, points=None, box=False, freq='W', con=con):

    _, weekly_data = get_ticker_data(symbol, con)

    if freq == 'W':

        ys = weekly_data

    for i, model in enumerate(models):
        predictions = get_from_db('Forecast', symbol, con, model_type=model)
        fittedvalues = predictions[(predictions.relevance == 2) & (predictions.price.notna())
                                   | (predictions.relevance == 1)]
        new_y = pd.DataFrame(fittedvalues.price).rename(columns={'price': model_fnames[i]})

        # to avoid InvalidIndexError
        ys = ys.loc[~ys.index.duplicated(keep='first')]
        new_y = new_y.loc[~new_y.index.duplicated(keep='first')]

        ys = pd.concat([ys, new_y], join='inner', axis=1)

    metrics_list = ['absolute_error', 'absolute_percentage_error', 'root_mean_square_error', 'max_error']

    residuals = All_residuals(ys, metrics_list, model_fnames)

    df = getattr(residuals, metric)
    df = df.stack().reset_index(level=1)

    if not df.empty:
        df.index = df.index.date

    df.rename(columns={df.columns[0]: 'model', df.columns[1]: 'error'}, inplace=True)

    residuals_fig = px.violin(df, y='model', x='error', color='model', points=points, box=box, hover_name=df.index,
                              color_discrete_sequence=px.colors.qualitative.Plotly[2:])

    if not box:

        residuals_fig.update_traces(side='positive', width=1.8)

    if metric == 'absolute_percentage_error':

        residuals_fig.update_layout(xaxis_title="Ошибка, %")

    else:

        residuals_fig.update_layout(xaxis_title="Ошибка")

    residuals_fig.update_layout(
        title="Распределение ошибок (KDE)",
        yaxis_title=None,
        legend_title="Модели",
        font=dict(
            family="Source Sans Pro",
            size=16
        )
    )

    return residuals_fig, residuals.metric_df


@st.experimental_memo(ttl=86400)  # 24 hours
def update_data_if_required(day, con=con):

    with open("./files/flags.json", "r") as read_file:
        json_data = json.load(read_file)

    last_history_update = datetime.strptime(json_data['last_history_update'], "%Y-%m-%d").date()
    next_forecasting = datetime.strptime(json_data['next_forecasting'], "%Y-%m-%d").date()

    if day > last_history_update:

        last_symbol = get_symbol_list()[-1]

        if get_from_db("History", last_symbol, con, start=day.strftime("%Y-%m-%d")).empty:

            update_history(con, start=json_data['last_history_update'], end=(day+timedelta(days=1)).strftime("%Y-%m-%d"))

    if day >= next_forecasting:

        max_recorded_forecast = pd.read_sql(
            'SELECT MAX(recorded) FROM Forecast', con, parse_dates={"MAX(recorded)": "%Y-%m-%d"}
        ).loc[0][0].date()

        if max_recorded_forecast < day:

            weekly_forecasting(con)


@st.experimental_singleton() 
def get_backtest_fig(symbol, model, con=con):

    data = get_data_for_bt(symbol, model, con)

    bt = Backtest(data, MidTermEstimation, 
                cash=10000, commission=.002,
                exclusive_orders=True)
    bt.run(con=con, value_factor=0.25)
    return bt.plot(open_browser = False)


# Beginning
yesterday = date.today() - timedelta(days=1)
update_data_if_required(yesterday)

symbol_list = get_symbol_list()
tickers = pd.read_sql('SELECT * FROM Ticker', con, index_col="symbol")

models = pd.read_sql('SELECT * FROM Model', con)

st.header("Выбор моделей прогнозирования цены биржевых товаров")

st.markdown("Описание проекта на [GitHub](https://github.com/PavelNikishin/price-forecasting#readme)")

st.subheader('Результат предсказания модели')

# Forecast
# wigets
forecast_filter = st.columns(2)

with forecast_filter[0]:
    selected_symbol_name = st.selectbox('Инструмент', tickers.fullName.values)

with forecast_filter[1]:
    selected_model_name = st.selectbox('Модель прогнозирования', models.full_name.values)

# parameters processing
selected_symbol = tickers[tickers.fullName == selected_symbol_name].index.tolist()[0]
selected_model = models[models.full_name == selected_model_name].short_name.values[0]


# plot
forecast_fig = get_forecast_fig(selected_symbol, selected_model)

st.plotly_chart(forecast_fig, use_container_width=True, theme=None)

st.markdown("___")

st.subheader('Сравнение моделей')

# Residuals
# wigets
residuals_filter = st.columns(2)

with residuals_filter[0]:
    residuals_symbol_name = st.selectbox('Инструмент', tickers.fullName.values, key='Instrument_residuals_filter')

    if st.checkbox("Показывать Boxplot"):
        show_boxplot = True
    else:
        show_boxplot = False

    if st.checkbox("Показывать все значения", value=True):
        show_all_points = True
    else:
        show_all_points = False

with residuals_filter[1]:
    residuals_model_names = st.multiselect('Модели прогнозирования', models.full_name.values, default=models.full_name.values)
    metric_name = st.radio("Показатель", ['Абсолютная ошибка', 'Абсолютная процентная ошибка'], index=1)

# parameters processing
residuals_symbol = tickers[tickers.fullName == residuals_symbol_name].index.tolist()[0]
residuals_models = models.short_name[models.full_name.isin(residuals_model_names)].to_list()


if metric_name == 'Абсолютная ошибка':

    metric = 'absolute_error'

if metric_name == 'Абсолютная процентная ошибка':

    metric = 'absolute_percentage_error'

residuals_extra_args = {}

if any([show_boxplot, show_all_points]):

    if show_boxplot:

        residuals_extra_args['box'] = True

    if show_all_points:

        residuals_extra_args['points'] = 'all'


# plot
residuals_fig, residuals_df = get_residuals_fig(residuals_symbol, residuals_model_names, residuals_models,
                                                metric=metric, **residuals_extra_args)

st.plotly_chart(residuals_fig, use_container_width=True, theme=None)

# st.markdown("___")

st.subheader('Метрики')

st.dataframe(residuals_df)

st.markdown("___")

st.subheader('Пример использования в торговой стратегии')

st.markdown("""
    Построим на основании прогноза модели простейшую среднесрочную стратегию. \
    Будем считать среднесрочным прогнозом среднее значение предсказания цены за следующие 3-6 месяцев.
    - Открываем позицию, если прогноз на 25% выше/ниже текущей цены
    - Закрываем, когда цена и прогноз сравнялись
    """)

# wigets
bt_filter = st.columns(2)

with bt_filter[0]:
    selected_bt_symbol_name = st.selectbox('Инструмент', tickers.fullName.values, key='bt_symbol_name')

with bt_filter[1]:
    selected_bt_model_name = st.selectbox('Модель прогнозирования', models.full_name.values, key='bt_model_name')

# parameters processing
selected_bt_symbol = tickers[tickers.fullName == selected_bt_symbol_name].index.tolist()[0]
selected_bt_model = models[models.full_name == selected_bt_model_name].short_name.values[0]

bt_plot = get_backtest_fig(selected_bt_symbol, selected_bt_model)

st.bokeh_chart(bt_plot)

st.markdown("""
    Результат такой замечательный, потому что мы по факту заглядываем в будущее. Чтобы этого избежать, нужно использовать данные на дату \
    прогноза. Сейчас результаты записываются на глубину в год, но данных пока недостаточно. За ‘учебный’ период данные получить можно, \
    но придется менять настройки алгоритмов, либо сделать симуляцию, прогоняя программу по всем расчетным датам. 
    """)