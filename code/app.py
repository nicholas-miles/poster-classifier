# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from season_rating import df_seasons    

df = df_seasons('tt0386676')

app = dash.Dash()

app.layout = html.Div(children=[
    html.Label('Dropdown'),

    # dcc.Dropdown(
    #     options=[df.season.unique]
    # ),

    html.Div
    dcc.Graph(
        id='season_graph',
        figure={
            'data': [
                go.Scatter(
                    x=df[df['season'] == i]['episode'],
                    y=df[df['season'] == i]['rating'],
                    text='Episode ' + str(df[df['season'] == i]['episode']),
                    mode='markers',
                    opacity=0.7,
                    marker={
                        'size': 15,
                        'line': {'width': 0.5, 'color': 'white'}
                    },
                    name=i
                ) for i in df.season.unique()
            ],
            'layout': go.Layout(
                xaxis={'title': 'Episode #'},
                yaxis={'title': 'Rating'},
                margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                legend={'x': 0, 'y': 1},
                hovermode='closest'
            )
        }
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)