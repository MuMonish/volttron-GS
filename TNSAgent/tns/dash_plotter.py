import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.express as px
import pickle

def create_dash_app(local_assets=None):
    options = [
        {'label': 'Gas Turbine 1', 'value': 'GT1'},
        {'label': 'Gas Turbine 2', 'value': 'GT2'},
        {'label': 'Gas Turbine 3', 'value': 'GT3'},
        {'label': 'Gas Turbine 4', 'value': 'GT4'},
    ]
    if local_assets is not None:
        options = []
        dfs = {}
        for asset in local_assets:
            options.append({'label': asset.name, 'value': asset.name})
            dfs[asset.name] = pd.read_csv('Outputs/' + asset.name + '_output.csv',
                                          names=['date', 'timestamp', '1', '2', '3'])

    app = dash.Dash()

    app.title = 'Dash Tutorial'

    app.layout = html.Div([
        html.H1([
            html.H1(children='Simulation Results'),

            html.Div(children='''
                Dash: A web application framework for Python.
            ''')]),
        dcc.Checklist(
            id='selector',
            options=options,
            labelStyle={'display': 'inline-block'}
        ),
        dcc.Graph(
            id='graph1'
        ),
        dcc.Graph(
            id='graph2'
        ),
        dcc.Graph(
            id='graph3'
        )

    ])

    @app.callback(
        dash.dependencies.Output('graph1', 'figure'),
        [dash.dependencies.Input('selector', 'value')])
    def update_graph_src(selector):
        data = []
        if selector is not None:
            for name in selector:
                data.append({'x': dfs[name]['date'], 'y': dfs[name]['1'], 'type': 'line', 'name': name})

        figure = {
            'data': data,
            'layout': {'title': 'Results Visualization'}
        }
        return figure

    @app.callback(
        dash.dependencies.Output('graph2', 'figure'),
        [dash.dependencies.Input('selector', 'value')])
    def update_graph_src(selector):
        data = []
        if selector is not None:
            for name in selector:
                data.append({'x': dfs[name]['date'], 'y': dfs[name]['2'], 'type': 'line', 'name': name})

        figure = {
            'data': data,
            'layout': {'title': 'Results Visualization'}
        }
        return figure

    @app.callback(
        dash.dependencies.Output('graph3', 'figure'),
        [dash.dependencies.Input('selector', 'value')])
    def update_graph_src(selector):
        data = []
        if selector is not None:
            for name in selector:
                data.append({'x': dfs[name]['date'], 'y': dfs[name]['3'], 'type': 'line', 'name': name})

        figure = {
            'data': data,
            'layout': {'title': 'Results Visualization'}
        }
        return figure
    return app

if __name__ == '__main__':
    gt1 = pd.read_csv('Outputs/GasTurbine1_output.csv', names=['date', 'timestamp', '1', '2', '3'])
    gt2 = pd.read_csv('Outputs/GasTurbine2_output.csv', names=['date', 'timestamp', '1', '2', '3'])
    gt3 = pd.read_csv('Outputs/GasTurbine3_output.csv', names=['date', 'timestamp', '1', '2', '3'])
    gt4 = pd.read_csv('Outputs/GasTurbine4_output.csv', names=['date', 'timestamp', '1', '2', '3'])
    dfs = {"GT1": gt1, "GT2": gt2, "GT3": gt3, "GT4": gt4}
    app = create_dash_app()
    app.run_server(debug=True)