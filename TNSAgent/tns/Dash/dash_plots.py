import os
import pathlib
import numpy as np
import datetime as dt
import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html

from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State

import pickle

with open("../Outputs/localAssets.pkl", "rb") as f:  # Outputs/localAssets.pkl
    localAssets = pickle.load(f)

case_description = "Base case. No flexible loads."

dbs = {}
boilers = {}
chillers = {}
gas_turbines = {}
inflexibles = {}
flexibles = {}
pvs = {}
boiler_options = []
gt_options = []
chiller_options = []
flex_options = []
inflex_options = []
pv_options = []

la = localAssets[0]
costs = pd.DataFrame(np.zeros(len(localAssets[0].name)))
totals = pd.DataFrame(np.zeros(len(localAssets[0].name)))
for i, la in enumerate(localAssets):
    name = la.model.name
    model_type = la.model.model_type
    dbs[name] = pd.read_csv("../Outputs/" + name + "_output.csv")
    if i == 0:  # only on first iteration get dimensions to create a costs df
        costs = pd.DataFrame({
            "elec": np.zeros(len(dbs[name].iloc[:, 0])),
            "cool": np.zeros(len(dbs[name].iloc[:, 0])),
            "heat": np.zeros(len(dbs[name].iloc[:, 0])),
            "grid": np.zeros(len(dbs[name].iloc[:, 0])),
        })
        totals = pd.DataFrame({
            'elec_load': np.zeros(len(dbs[name].iloc[:, 0])),
            'elec_source': np.zeros(len(dbs[name].iloc[:, 0])),
            'cool_load': np.zeros(len(dbs[name].iloc[:, 0])),
            'cool_source': np.zeros(len(dbs[name].iloc[:, 0])),
            'heat_load': np.zeros(len(dbs[name].iloc[:, 0])),
            'heat_source': np.zeros(len(dbs[name].iloc[:, 0])),
            'gas': np.zeros(len(dbs[name].iloc[:, 0])),
            'cost': np.zeros(len(dbs[name].iloc[:, 0])),
        })
    if model_type in ['InflexibleBuilding']:
        inflexibles[name] = dbs[name]
        inflex_options.append({'label': name, 'value': name})
        totals["elec_load"] += inflexibles[name]['Electricity Dispatched']
        totals["heat_load"] += inflexibles[name]['Heat Dispatched']
        totals["cool_load"] += inflexibles[name]['Cool Dispatched']
    if model_type in ['FlexibleBuilding']:
        flexibles[name] = dbs[name]
        flex_options.append({'label': name, 'value': name})
        totals["elec_load"] += flexibles[name]['Electricity Dispatched']
        totals["heat_load"] += flexibles[name]['Heat Dispatched']
        totals["cool_load"] += flexibles[name]['Cool Dispatched']
    if model_type in ['Boiler']:
        boilers[name] = dbs[name]
        boiler_options.append({'label': name, 'value': name})
        costs["heat"] += dbs[name]["Cost"]
        totals["heat_source"] += boilers[name]['Heat Dispatched']
        totals["gas"] += boilers[name]['Gas Consumed']
        totals["cost"] += boilers[name]['Cost']
    if model_type in ['GasTurbine']:
        gas_turbines[name] = dbs[name]
        gt_options.append({'label': name, 'value': name})
        costs["elec"] += dbs[name]["Cost"]
        totals["elec_source"] += gas_turbines[name]['Electricity Dispatched']
        totals["gas"] += gas_turbines[name]['Gas Consumed']
        totals["cost"] += gas_turbines[name]['Cost']
    if model_type in ['Chiller']:
        chillers[name] = dbs[name]
        chiller_options.append({'label': name, 'value': name})
        costs["cool"] += dbs[name]["Cost"]
        totals["cool_source"] += chillers[name]['Cool Dispatched']
        totals["elec_load"] += chillers[name]['Electricity Consumed']
        totals["cost"] += chillers[name]['Cost']
    if model_type in ['SolarPV']:
        pvs[name] = dbs[name]
        pv_options.append({'label': name, 'value': name})
        totals["elec_source"] += pvs[name]['Electricity Dispatched']

solution_cost = pd.read_csv("../Outputs/solutions_output.csv")
costs["grid"] += solution_cost["Grid Cost"]
totals['elec_source'] += solution_cost['Grid Power']
totals['TimeStamp'] = solution_cost["TimeStamp"]

app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

server = app.server

# app_color = {"graph_bg": "#efebe1", "graph_line": "#1e3749"}
app_color = {"graph_bg": "#ffffff", "graph_line": "#1e3749"}

app.layout = html.Div(
    [
        # header
        html.Div(
            [
                html.Div(
                    [
                        html.H4("SIMULATION RESULTS", className="app__header__title"),
                        html.P(
                            "CASE DESCRIPTION: " + case_description,
                            className="app__header__title--grey",
                        ),
                    ],
                    className="app__header__desc",
                ),
            ],
            className="app__header",
        ),
        html.Div(
            [
                # Graphs
                html.Div(
                    [
                        # Graph 1 (Total Costs)
                        html.Div(
                            [  # Graph 1 Title
                                html.Div(
                                    [html.H6("Solution Cost ($)", className="graph__title")], className="content"),
                                dcc.Graph(
                                    id="sol-cost",
                                    figure=dict(
                                        layout=dict(
                                            plot_bgcolor=app_color["graph_bg"],
                                            paper_bgcolor=app_color["graph_bg"],
                                            height=400,
                                            xaxis={
                                                "showline": True,
                                                "zeroline": False,
                                                "title": "Time",
                                            },
                                            yaxis={
                                                "showgrid": True,
                                                "showline": True,
                                                "fixedrange": True,
                                                "zeroline": True,
                                                "gridcolor": app_color["graph_line"],
                                            },
                                        ),
                                        data=[
                                            dict(
                                                name="Utility Cost",
                                                type="scatter",
                                                mode="lines",
                                                y=costs["grid"],
                                                x=solution_cost["TimeStamp"],
                                            ),
                                            dict(
                                                name="Electric Generation Cost",
                                                type="scatter",
                                                mode="lines",
                                                y=costs["elec"],
                                                x=solution_cost["TimeStamp"],
                                            ),
                                            dict(
                                                name="Heating Cost",
                                                type="scatter",
                                                mode="lines",
                                                y=costs["heat"],
                                                x=solution_cost["TimeStamp"],
                                            )
                                        ]
                                    ),
                                ),

                            ],
                            className="graph__container first",
                        ),
                        # Graph 2 (Electricity)
                        html.Div(
                            [  # Graph 2 Title
                                html.Div(
                                    [html.H6("Electricity Dispatched (kW)", className="two-thirds graph__title"),
                                     html.Div(
                                         [  # Checklist
                                             dcc.Checklist(
                                                 id='elec_total_check',
                                                 options=[{'label': "Total Source", 'value': 'elec_source'},
                                                          {'label': "Total Load", 'value': 'elec_load'}],
                                                 inputClassName="auto__checkbox",
                                                 labelClassName="auto__label",
                                                 labelStyle={'display': 'inline-block'}
                                             ),
                                         ], className="one-sixth graph__check")
                                     ], className="content"
                                ),
                                dcc.Graph(
                                    id="elec-disp",
                                    figure=dict(
                                        layout=dict(
                                            plot_bgcolor=app_color["graph_bg"],
                                            paper_bgcolor=app_color["graph_bg"],
                                        )
                                    ),
                                ),

                            ],
                            className="graph__container middle",
                        ),
                        # Graph 3 (Heat)
                        html.Div(
                            [
                                html.Div(
                                    [html.H6("Heat Dispatched ()", className="graph__title"),
                                     html.Div(
                                         [  # Checklist
                                             dcc.Checklist(
                                                 id='heat_total_check',
                                                 options=[{'label': "Total Source", 'value': 'heat_source'},
                                                          {'label': "Total Load", 'value': 'heat_load'}],
                                                 inputClassName="auto__checkbox",
                                                 labelClassName="auto__label",
                                                 labelStyle={'display': 'inline-block'}
                                             ),
                                         ], className="one-sixth graph__check")
                                     ], className="content"),
                                dcc.Graph(
                                    id="heat-disp",
                                    figure=dict(
                                        layout=dict(
                                            plot_bgcolor=app_color["graph_bg"],
                                            paper_bgcolor=app_color["graph_bg"],
                                        )
                                    ),
                                ),
                            ],
                            className="graph__container middle",
                        ),
                        # Graph 4 (Cooling)
                        html.Div(
                            [
                                html.Div(
                                    [html.H6("Cooling Dispatched ()", className="graph__title"),
                                     html.Div(
                                         [  # Checklist
                                             dcc.Checklist(
                                                 id='cool_total_check',
                                                 options=[{'label': "Total Source", 'value': 'cool_source'},
                                                          {'label': "Total Load", 'value': 'cool_load'}],
                                                 inputClassName="auto__checkbox",
                                                 labelClassName="auto__label",
                                                 labelStyle={'display': 'inline-block'}
                                             ),
                                         ], className="one-sixth graph__check")
                                     ],
                                    className="content"
                                ),
                                dcc.Graph(
                                    id="cool-disp",
                                    figure=dict(
                                        layout=dict(
                                            plot_bgcolor=app_color["graph_bg"],
                                            paper_bgcolor=app_color["graph_bg"],
                                        )
                                    ),
                                ),
                            ],
                            className="graph__container middle",
                        ),
                        # Graph 5 (Gas)
                        html.Div(
                            [
                                html.Div(
                                    [html.H6("Fuel Consumed ()", className="graph__title"),
                                     html.Div(
                                         [  # Checklist
                                             dcc.Checklist(
                                                 id='gas_total_check',
                                                 options=[{'label': "Total", 'value': 'gas'}],
                                                 inputClassName="auto__checkbox",
                                                 labelClassName="auto__label",
                                                 labelStyle={'display': 'inline-block'}
                                             ),
                                         ], className="one-sixth graph__check")
                                     ], className="content"
                                ),
                                dcc.Graph(
                                    id="fuel-cons",
                                    figure=dict(
                                        layout=dict(
                                            plot_bgcolor=app_color["graph_bg"],
                                            paper_bgcolor=app_color["graph_bg"],
                                        )
                                    ),
                                ),
                            ],
                            className="graph__container middle",
                        ),
                        # Graph 6 (Individual Costs)
                        html.Div(
                            [
                                html.Div(
                                    [html.H6("Cost ($)", className="graph__title")], className="content"
                                ),
                                dcc.Graph(
                                    id="cost",
                                    figure=dict(
                                        layout=dict(
                                            plot_bgcolor=app_color["graph_bg"],
                                            paper_bgcolor=app_color["graph_bg"],
                                        )
                                    ),
                                ),
                            ],
                            className="graph__container second",
                        ),
                    ],
                    className="two-thirds column",
                ),
                html.Div([], className="one-third column"),  # empty container to make space for menu
                html.Div(
                    [
                        # Menu
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.H6(
                                            "LOCAL ASSETS",
                                            className="graph__title",
                                        )
                                    ]
                                ),
                                html.Div(
                                    [
                                        html.Div(
                                            [html.H6("Gas Turbines", className="graph__title")]
                                        ),
                                        # Checklist
                                        dcc.Checklist(
                                            id='gt_checklist',
                                            options=gt_options,
                                            inputClassName="auto__checkbox",
                                            labelClassName="auto__label",
                                            labelStyle={'display': 'inline-block'}
                                        ),
                                    ]
                                ),
                                html.Div(
                                    [
                                        html.Div(
                                            [html.H6("Solar PV", className="graph__title")]
                                        ),
                                        # Checklist
                                        dcc.Checklist(
                                            id='pv_checklist',
                                            options=pv_options,
                                            inputClassName="auto__checkbox",
                                            labelClassName="auto__label",
                                            labelStyle={'display': 'inline-block'}
                                        ),
                                    ]
                                ),
                                html.Div(
                                    [
                                        html.Div(
                                            [html.H6("Chillers", className="graph__title")]
                                        ),
                                        # Checklist
                                        dcc.Checklist(
                                            id='chiller_checklist',
                                            options=chiller_options,
                                            inputClassName="auto__checkbox",
                                            labelClassName="auto__label",
                                            labelStyle={'display': 'inline-block'}
                                        ),
                                    ]
                                ),
                                html.Div(
                                    [
                                        html.Div(
                                            [html.H6("Boilers", className="graph__title")]
                                        ),
                                        # Checklist
                                        dcc.Checklist(
                                            id='boiler_checklist',
                                            options=boiler_options,
                                            inputClassName="auto__checkbox",
                                            labelClassName="auto__label",
                                            labelStyle={'display': 'inline-block'}
                                        ),
                                    ]
                                ),
                                html.Div(
                                    [
                                        html.Div(
                                            [html.H6("Inflexible Loads", className="graph__title")]
                                        ),
                                        # Checklist
                                        dcc.Checklist(
                                            id='inflex_checklist',
                                            options=inflex_options,
                                            inputClassName="auto__checkbox",
                                            labelClassName="auto__label",
                                            labelStyle={'display': 'inline-block'}
                                        ),
                                    ]
                                ),
                                html.Div(
                                    [
                                        html.Div(
                                            [html.H6("Flexible Loads", className="graph__title")]
                                        ),
                                        # Checklist
                                        dcc.Checklist(
                                            id='flex_checklist',
                                            options=flex_options,
                                            inputClassName="auto__checkbox",
                                            labelClassName="auto__label",
                                            labelStyle={'display': 'inline-block'}
                                        ),
                                    ]
                                ),
                            ],
                            className="graph__container first",
                        ),
                    ],
                    className="one-third column fixed",
                ),
            ],
            className="app__content",
        ),
    ],
    className="app__container",
)


@app.callback(
    Output("elec-disp", "figure"),
    [
        Input("gt_checklist", "value"),
        Input("pv_checklist", "value"),
        Input("chiller_checklist", "value"),
        Input("inflex_checklist", "value"),
        Input("flex_checklist", "value"),
        Input("elec_total_check", "value")
    ]
)
def elec_graph(gt_selector, pv_selector, chiller_selector, inflex_selector, flex_selector, total_selector):
    data = []
    total = pd.DataFrame(np.zeros(len(gas_turbines["GasTurbine1"]["TimeStamp"])))
    if gt_selector is not None:
        for name in gt_selector:
            total[0] = total[0] + gas_turbines[name]["Electricity Dispatched"]
            trace = dict(
                name=name,
                type="scatter",
                # line={"color": "#42C4F7"},
                mode="lines",
                y=gas_turbines[name]["Electricity Dispatched"],
                x=gas_turbines[name]["TimeStamp"],
            )
            data.append(trace)
    if pv_selector is not None:
        for name in pv_selector:
            total[0] = total[0] + pvs[name]["Electricity Dispatched"]
            trace = dict(
                name=name,
                type="scatter",
                # line={"color": "#42C4F7"},
                mode="lines",
                y=pvs[name]["Electricity Dispatched"],
                x=pvs[name]["TimeStamp"],
            )
            data.append(trace)
    if chiller_selector is not None:
        for name in chiller_selector:
            total[0] = total[0] + chillers[name]["Electricity Consumed"]
            trace = dict(
                name=name,
                type="scatter",
                # line={"color": "#42C4F7"},
                mode="lines",
                y=chillers[name]["Electricity Consumed"],
                x=chillers[name]["TimeStamp"],
            )
            data.append(trace)
    if inflex_selector is not None:
        for name in inflex_selector:
            trace = dict(
                name=name,
                type="scatter",
                mode="lines",
                y=inflexibles[name]["Electricity Dispatched"],
                x=inflexibles[name]["TimeStamp"],
            )
            data.append(trace)
    if flex_selector is not None:
        for name in flex_selector:
            total[0] = total[0] + flexibles[name]["Electricity Dispatched"]
            trace = dict(
                name=name,
                type="scatter",
                # line={"color": "#42C4F7"},
                mode="lines",
                y=flexibles[name]["Electricity Dispatched"],
                x=flexibles[name]["TimeStamp"],
            )
            data.append(trace)
    if total_selector is not None:
        for name in total_selector:
            if "source" in name:
                label = "Total Source"
            else:
                label = "Total Load"
            trace = dict(
                name=label,
                type="scatter",
                mode="lines",
                y=totals[name],
                x=totals["TimeStamp"],
            )
            data.append(trace)

    layout = dict(
        plot_bgcolor=app_color["graph_bg"],
        paper_bgcolor=app_color["graph_bg"],
        # font={"color": "#000"},
        height=400,
        xaxis={
            "showline": True,
            "zeroline": False,
            "title": "Time",
        },
        yaxis={
            "showgrid": True,
            "showline": True,
            "fixedrange": True,
            "zeroline": True,
            "gridcolor": app_color["graph_line"],
        },
    )
    return dict(data=data, layout=layout)


@app.callback(
    Output("heat-disp", "figure"),
    [
        Input("boiler_checklist", "value"),
        Input("inflex_checklist", "value"),
        Input("flex_checklist", "value"),
        Input("heat_total_check", "value")
    ]
)
def heat_graph(boiler_selector, inflex_selector, flex_selector, total_selector):
    data = []
    if boiler_selector is not None:
        for name in boiler_selector:
            trace = dict(
                name=name,
                type="scatter",
                mode="lines",
                y=boilers[name]["Heat Dispatched"],
                x=boilers[name]["TimeStamp"],
            )
            data.append(trace)
    if inflex_selector is not None:
        for name in inflex_selector:
            trace = dict(
                name=name,
                type="scatter",
                mode="lines",
                y=inflexibles[name]["Heat Dispatched"],
                x=inflexibles[name]["TimeStamp"],
            )
            data.append(trace)
    if flex_selector is not None:
        for name in flex_selector:
            trace = dict(
                name=name,
                type="scatter",
                mode="lines",
                y=flexibles[name]["Heat Dispatched"],
                x=flexibles[name]["TimeStamp"],
            )
            data.append(trace)
    if total_selector is not None:
        for name in total_selector:
            if "source" in name:
                label = "Total Source"
            else:
                label = "Total Load"
            trace = dict(
                name=label,
                type="scatter",
                mode="lines",
                y=totals[name],
                x=totals["TimeStamp"],
            )
            data.append(trace)
    layout = dict(
        plot_bgcolor=app_color["graph_bg"],
        paper_bgcolor=app_color["graph_bg"],
        # font={"color": "#0"},
        height=400,
        xaxis={
            "showline": True,
            "zeroline": False,
            "title": "Time",
        },
        yaxis={
            "showgrid": True,
            "showline": True,
            "fixedrange": True,
            "zeroline": True,
            "gridcolor": app_color["graph_line"],
        },
    )
    return dict(data=data, layout=layout)


@app.callback(
    Output("cool-disp", "figure"),
    [
        Input("chiller_checklist", "value"),
        Input("inflex_checklist", "value"),
        Input("flex_checklist", "value"),
        Input("cool_total_check", "value")
    ]
)
def cool_graph(chiller_selector, inflex_selector, flex_selector, total_selector):
    data = []
    if chiller_selector is not None:
        for name in chiller_selector:
            trace = dict(
                name=name,
                type="scatter",
                mode="lines",
                y=chillers[name]["Cool Dispatched"],
                x=chillers[name]["TimeStamp"],
            )
            data.append(trace)
    if inflex_selector is not None:
        for name in inflex_selector:
            trace = dict(
                name=name,
                type="scatter",
                mode="lines",
                y=inflexibles[name]["Cool Dispatched"],
                x=inflexibles[name]["TimeStamp"],
            )
            data.append(trace)
    if flex_selector is not None:
        for name in flex_selector:
            trace = dict(
                name=name,
                type="scatter",
                mode="lines",
                y=flexibles[name]["Cool Dispatched"],
                x=flexibles[name]["TimeStamp"],
            )
            data.append(trace)
    if total_selector is not None:
        for name in total_selector:
            if "source" in name:
                label = "Total Source"
            else:
                label = "Total Load"
            trace = dict(
                name=label,
                type="scatter",
                mode="lines",
                y=totals[name],
                x=totals["TimeStamp"],
            )
            data.append(trace)
    layout = dict(
        plot_bgcolor=app_color["graph_bg"],
        paper_bgcolor=app_color["graph_bg"],
        # font={"color": "#0"},
        height=400,
        xaxis={
            "showline": True,
            "zeroline": False,
            "title": "Time",
        },
        yaxis={
            "showgrid": True,
            "showline": True,
            "fixedrange": True,
            "zeroline": True,
            "gridcolor": app_color["graph_line"],
        },
    )
    return dict(data=data, layout=layout)


@app.callback(
    Output("fuel-cons", "figure"),
    [
        Input("gt_checklist", "value"),
        Input("boiler_checklist", "value"),
        Input("gas_total_check", "value")
    ]
)
def gas_graph(gt_selector, boiler_selector, total_selector):
    data = []
    if gt_selector is not None:
        for name in gt_selector:
            trace = dict(
                name=name,
                type="scatter",
                # line={"color": "#42C4F7"},
                mode="lines",
                y=gas_turbines[name]["Gas Consumed"],
                x=gas_turbines[name]["TimeStamp"],
            )
            data.append(trace)
    if boiler_selector is not None:
        for name in boiler_selector:
            trace = dict(
                name=name,
                type="scatter",
                # line={"color": "#42C4F7"},
                mode="lines",
                y=boilers[name]["Gas Consumed"],
                x=boilers[name]["TimeStamp"],
            )
            data.append(trace)
    if total_selector is not None:
        for name in total_selector:
            trace = dict(
                name="Total",
                type="scatter",
                mode="lines",
                y=totals[name],
                x=totals["TimeStamp"],
            )
            data.append(trace)
    layout = dict(
        plot_bgcolor=app_color["graph_bg"],
        paper_bgcolor=app_color["graph_bg"],
        # font={"color": "#0"},
        height=400,
        xaxis={
            "showline": True,
            "zeroline": False,
            "title": "Time",
        },
        yaxis={
            "showgrid": True,
            "showline": True,
            "fixedrange": True,
            "zeroline": True,
            "gridcolor": app_color["graph_line"],
        },
    )
    return dict(data=data, layout=layout)


@app.callback(
    Output("cost", "figure"),
    [
        Input("gt_checklist", "value"),
        Input("chiller_checklist", "value"),
        Input("boiler_checklist", "value")
    ]
)
def cost_graph(gt_selector, chiller_selector, boiler_selector):
    data = []
    if gt_selector is not None:
        for name in gt_selector:
            trace = dict(
                name=name,
                type="scatter",
                # line={"color": "#42C4F7"},
                mode="lines",
                y=gas_turbines[name]["Cost"],
                x=gas_turbines[name]["TimeStamp"],
            )
            data.append(trace)
    if chiller_selector is not None:
        for name in chiller_selector:
            trace = dict(
                name=name,
                type="scatter",
                # line={"color": "#42C4F7"},
                mode="lines",
                y=chillers[name]["Cost"],
                x=chillers[name]["TimeStamp"],
            )
            data.append(trace)
    if boiler_selector is not None:
        for name in boiler_selector:
            trace = dict(
                name=name,
                type="scatter",
                # line={"color": "#42C4F7"},
                mode="lines",
                y=boilers[name]["Cost"],
                x=boilers[name]["TimeStamp"],
            )
            data.append(trace)
    layout = dict(
        plot_bgcolor=app_color["graph_bg"],
        paper_bgcolor=app_color["graph_bg"],
        # font={"color": "#fff"},
        height=400,
        xaxis={
            "showline": True,
            "zeroline": False,
            "title": "Time",
        },
        yaxis={
            "showgrid": True,
            "showline": True,
            "fixedrange": True,
            "zeroline": True,
            "gridcolor": app_color["graph_line"],
        },
    )
    return dict(data=data, layout=layout)


if __name__ == "__main__":
    app.run_server(debug=True)
