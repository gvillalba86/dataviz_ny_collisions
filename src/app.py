from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
import pandas as pd
from pathlib import Path
import calendar
from collections import Counter
from ast import literal_eval
import dash_bootstrap_components as dbc
import json
import dash_daq as daq
import requests


path = Path('data') / 'cleaned_nyc_crashes.csv'
url = 'https://gerson-bucket-01.s3.eu-south-2.amazonaws.com/cleaned_nyc_crashes.csv'
df = pd.read_csv(url, parse_dates=['DATETIME'], converters={"FACTORS": literal_eval, "VEHICLE TYPES": literal_eval})
df.columns = [col.capitalize() for col in df.columns]
df.rename(columns={'N vehicles': 'Vehicles involved'}, inplace=True)
df['Total crashes'] = 1

years = sorted(df['Datetime'].dt.year.unique().tolist(), reverse=True)
boroughs = df['Borough'].dropna().unique().tolist()

# Load GeoJSONs
url = 'https://gerson-bucket-01.s3.eu-south-2.amazonaws.com/new-york-city-boroughs.geojson'
response = requests.request("GET", url)
if response.status_code == 200:
    nyc_boroughs = response.json()

url = 'https://gerson-bucket-01.s3.eu-south-2.amazonaws.com/nyc-neighborhoods.geo.json'
response = requests.request("GET", url)
if response.status_code == 200:
    nyc_districts = response.json()

app = Dash(__name__,  external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

borough_colors = ["#ef476f","#ffd166","#f78c6b", "#06d6a0","#118ab2"]

subtitle_text = ''' 
This data-driven visualization study explores car accidents in New York City, aiming to uncover meaningful 
insights and patterns in accident data. By analyzing various contributing factors, such as time, location and vehicle types,
the research seeks to inform evidence-based policy decisions and targeted traffic safety initiatives. Through visually 
representing the data, the study aims to raise public awareness, promote responsible driving, and foster a safer urban 
environment. The findings provide stakeholders and policymakers with valuable insights to allocate resources effectively 
and develop interventions that mitigate risks and enhance traffic safety measures, ultimately setting an example for urban road safety worldwide.
'''

def create_victims_graph():
    cols = [col for col in df.columns if ('killed' in col.lower() or 'injured' in col.lower()) and 'person' not in col.lower()]
    dff = df[cols].sum(axis=0).to_frame().reset_index().rename(columns={'index': 'Victim', 0: 'Total'})
    dff['Seriousness'] = dff['Victim'].apply(lambda x: 'Killed' if 'killed' in x.lower() else 'Injured')
    dff['Victim'] = dff['Victim'].apply(lambda x: x.replace('killed', '').replace('injured', '').strip().title())
    fig_killed = px.pie(dff[dff['Seriousness']=='Killed'], names='Victim', values='Total')
    fig_injured = px.pie(dff[dff['Seriousness']=='Injured'], names='Victim', values='Total')
    for fig in [fig_injured, fig_killed]:
        fig.update_layout(
            autosize=True,
            margin={'l':0, 'r':0, 'b':0, 't':10, 'pad':0},
            coloraxis_showscale=False,
            legend_font_color= '#7fafdf',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation="h"),
        )
        fig.update_xaxes(title='', visible=True, showticklabels=True, color = '#7fafdf', gridcolor = '#48556e', showgrid=False)
        fig.update_yaxes(title='', visible=True, showticklabels=True, color = '#7fafdf', gridcolor = '#48556e', showgrid=False)
    return fig_injured, fig_killed

        
# Graphs          
choropleth_map = dcc.Graph(id = 'choropleth_map', config = {'displayModeBar': False}, className = 'map-height')
evol_graph_neighborhood = dcc.Graph(id = 'evol_graph_neighborhood', config = {'displayModeBar': False})
per_graph_neighborhood = dcc.Graph(id = 'per_graph_neighborhood', config = {'displayModeBar': False})


treemap = dcc.Graph(id = 'treemap', config = {'displayModeBar': False})
heatmap = dcc.Graph(id = 'heatmap', config = {'displayModeBar': False}, className = 'second-row')
evol_graph = dcc.Graph(id = 'evol_graph', config = {'displayModeBar': False}, className = 'second-row')

injured_fig, killed_fig = create_victims_graph()
injured_graph = dcc.Graph(figure = injured_fig, id = 'injured_graph', config = {'displayModeBar': False}, className = 'second-row')
killed_graph = dcc.Graph(figure = killed_fig, id = 'killed_graph', config = {'displayModeBar': False}, className = 'second-row')

factors_graph = dcc.Graph(id = 'factors_graph', config = {'displayModeBar': False}, className = 'second-row')
vehices_graph = dcc.Graph(id = 'vehices_graph', config = {'displayModeBar': False}, className = 'second-row')

# Cards
borough_neigborhood_card = dbc.Card([
    html.H2(id='selected-borough-name', className="card-title"),
    html.H4(id='selected-neighborhood-name', className="card-text"),
])

kpis_card = [
    dbc.Card([
        html.H2(id='num-crashes', className="card-title"),
        html.H4('Crashes', className="card-text"),
    ]),
    dbc.Card([
        html.H2(id='num-vehicles', className="card-title"),
        html.H4('Vehicles involved', className="card-text"),
    ]),
    dbc.Card(id='injured-card', children=[
        html.H2(id='num-injured', className="card-title"),
        html.H4('Injured', className="card-text"),
        html.H5(id='per-injured', className="card-title right-align"),
    ]),
    dbc.Card(id='killed-card',children=[
        html.H2(id='num-killed', className="card-title"),
        html.H4('Killed', className="card-text"),
        html.H5(id='per-killed', className="card-title right-align"),
    ]),
]

agg_cols = ['Total crashes', 'Vehicles involved'] + [col for col in df.columns if ('injured' in col.lower() or 'killed' in col.lower())]


# Layout
app.layout = html.Div(children=[
    # Header
    html.Div(id='header', className = 'header', children=[
        html.H1(children='NYC Crashes', className = 'title'),
        html.P(children=subtitle_text, className = 'description'),
    ]),
    dbc.Row(id = 'geo-filters', class_name='filter-row', children=[
        dbc.Col([
            html.P('Drag the slider to filter by years.'),
            dcc.RangeSlider(years[-1], years[0], 1, value=[years[-1], years[0]], id='slider-years', className = 'range-slider', 
                            marks={str(year): str(year) for year in years}
                ),
        ], width=6),
        dbc.Col([
            html.P('Use the dropdown to select a metric.'),
            dcc.Dropdown(options=agg_cols, value='Total crashes', id='metric-selector', clearable=False),
        ], width=3),
        dbc.Col([
            html.P('Select timeframe.'),
            dcc.Dropdown(
                options=['Year', 'Month', 'Day of week', 'Hour'],
                value='Year',
                id='timeframe-selector',
                clearable=False
            ),
        ], width=3),
    ]),
    # First row
    dbc.Row(id='row-1', class_name='pretty_container h-50', children=[
        dbc.Row(id = 'geo-distribution-header', className = 'graph-header', children=[
            html.H2('Geographical distribution of crashes', className = 'graph-title'),
        ]),
        dbc.Row([
            dbc.Col([
                choropleth_map,
            ], width=8),
            dbc.Col([
                dbc.Row(evol_graph_neighborhood),
                dbc.Row(per_graph_neighborhood),
            ], width=4),
        ]),
        dbc.Row([
            treemap,
        ])
    ]),
    # Second row
    dbc.Row(id='row-2', children=[
        dbc.Col(id='left-column-2', class_name='pretty_container h-25', width=5, children=[
            html.Div(id = 'heatmap-header', className = 'graph-header', children=[
                html.H2('Month and day of the week of occurrence', className = 'graph-title'),
            ]),
            heatmap
        ]),
        dbc.Col(id='right-column-2', class_name='pretty_container h-25', width=7, children=[
            dbc.Row(id = 'evolution-header', class_name = 'graph-header', children=[
                dbc.Col([
                    html.H2('Time trend', className = 'graph-title'),
                ], width=8),
                dbc.Col([
                    daq.ToggleSwitch(id='my-toggle-switch', value=True, label='Show by borough', labelPosition='bottom'),
                ], width=4),
            ]),
            dbc.Row(evol_graph),
            
        ]),
    ]),
    # Third row
    dbc.Row(id='row-3', children=[
        dbc.Col(id='right-column-3', class_name='pretty_container', width= 8, children=[
            dbc.Row([
                html.Div(id = 'graph3-header', className = 'graph-header', children=[
                    html.H2('Contributing factors and vehicle types involved', className = 'graph-title'),
                ]),
            ]),
            dbc.Row([
                dbc.Col(dcc.Dropdown(
                    options=boroughs,
                    value=boroughs,
                    id='borough-selector',
                    clearable=False,
                    multi=True             
                ), width=3),
                dbc.Col([
                    dcc.Tabs(id = "factos-vehicle-tabs", value = 'contributing-factors', children = [
                        dcc.Tab(factors_graph,
                                label = 'Contributing factors',
                                value = 'contributing-factors',
                                className = 'custom-tab',
                                selected_className = 'custom-tab-selected'),
                        dcc.Tab(vehices_graph,
                                label = 'Vehicle types involved',
                                value = 'vehicle-types',
                                className = 'custom-tab',
                                selected_className = 'custom-tab-selected'),
                    ]),
                ], width=9),
            ])
        ]),
        dbc.Col(id='left-column-3', class_name='pretty_container', width= 4, children=[
            dbc.Row([
                html.Div(id = 'graph1-header', className = 'graph-header', children=[
                    html.H2('Victim types', className = 'graph-title'),
                ]),
            ]),
            dbc.Row([
                dcc.Tabs(id = "victim-types-tabs", value = 'Killed', children = [
                    dcc.Tab(killed_graph,
                            label = 'Killed',
                            value = 'Killed',
                            className = 'custom-tab',
                            selected_className = 'custom-tab-selected'),
                    dcc.Tab(injured_graph,
                            label = 'Injured',
                            value = 'Injured',
                            className = 'custom-tab',
                            selected_className = 'custom-tab-selected'),
                ]),
            ]),
        ]),
    ]),
    dbc.Row(id='row-4', children=[
        html.H4('Sources:'),
        dcc.Link('NYC Open Data', href='https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95'),
        dcc.Link('NYC Gov', href='https://www.nyc.gov/site/planning/data-maps/open-data/districts-download-metadata.page'),
    ]),
], id="mainContainer", style={"display": "flex", "flex-direction": "column"})

    
    
@callback(
    Output('choropleth_map', 'figure'),
    [Input('slider-years', 'value'),
    Input('metric-selector', 'value')]
)
def update_map(years, stat_name):
    years_list = list(range(years[0], years[1]+1))
    dff = df[df['Datetime'].dt.year.isin(years_list)]
    hover_data = ['Borough', 'Neighborhood', 'Total crashes']
    
    geo_df = dff.groupby(['Borough', 'District'])[agg_cols].sum()
    geo_df = geo_df.astype(int).reset_index().rename(columns={'District': 'Neighborhood'})
    geo_df.columns = [col.capitalize() for col in geo_df.columns]

    fig = px.choropleth_mapbox(
        geo_df,
        geojson=nyc_districts,
        locations='Neighborhood',
        featureidkey='properties.name',
        hover_data=hover_data,
        color=stat_name,
        color_continuous_scale='Darkmint',
        mapbox_style='carto-positron',
        zoom=9.5, center={'lat': 40.7, 'lon': -73.95},
        opacity=1,
        labels={'Count': 'Number of crashes'},
    )
    fig.update_layout(
        autosize=True,
        margin={'l':0, 'r':0, 'b':0, 't':0, 'pad':0},
        showlegend=False,
        legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="left", x=0),
    )
    fig.update(layout_coloraxis_showscale=False)
    
    return fig

'''
@callback(
    [Output('selected-borough-name', 'children'),
     Output('selected-neighborhood-name', 'children'),
     Output('num-crashes', 'children'),
     Output('num-vehicles', 'children'),
     Output('num-injured', 'children'),
     Output('per-injured', 'children'),
     Output('num-killed', 'children'),
     Output('per-killed', 'children'),
     ],
    [Input('slider-years', 'value'),
    Input('choropleth_map', 'clickData')]
)
def update_stats(years, sel_region):
    years_list = list(range(years[0], years[1]+1))
    dff = df[df['Datetime'].dt.year.isin(years_list)]
    if sel_region is None:
        borough_name = 'New York City'
        neighborhood_name = 'All neighborhoods'
    else:
        borough_name = sel_region['points'][0]['customdata'][0]
        neighborhood_name = sel_region['points'][0]['customdata'][1]
        dff = dff[dff['District']==sel_region['points'][0]['customdata'][1]]
    return (borough_name, 
            neighborhood_name, 
            dff.shape[0], 
            dff['Vehicles involved'].sum(), 
            dff['PERSONS INJURED'].sum(), 
            f"{dff['PERSONS INJURED'].sum()/dff.shape[0]:.2%} of total",
            dff['PERSONS KILLED'].sum(),
            f"{dff['PERSONS KILLED'].sum()/dff.shape[0]:.2%} of total",
    )
'''

@callback(
    [Output('evol_graph_neighborhood', 'figure'),
    Output('per_graph_neighborhood', 'figure')],
    [Input('timeframe-selector', 'value'), 
     Input('slider-years', 'value'),
     Input('choropleth_map', 'clickData')]
)
def update_evolution_graph_neighborhood(timeframe, years, sel_region):
    years_list = list(range(years[0], years[1]+1))
    df_filtered = df[df['Datetime'].dt.year.isin(years_list)]
    if sel_region is not None:
        neighborhood_name = sel_region['points'][0]['customdata'][1]
        df_filtered = df_filtered[df_filtered['District'] == neighborhood_name]

    colors = {
        'Total crashes' : '#666b6a',
        'Vehicles involved' : '#afd0d6',
        'Persons injured' : '#fcab10',
        'Persons killed' : '#f8333c',
    }

    if timeframe == 'Year':
        filter_tf = df_filtered['Datetime'].dt.year
    elif timeframe == 'Month':
        filter_tf = df_filtered['Datetime'].dt.month
    elif timeframe == 'Day of week':
        filter_tf = df_filtered['Datetime'].dt.day_of_week
    else:
        filter_tf = df_filtered['Datetime'].dt.hour
    dff = df_filtered.groupby(filter_tf)[agg_cols].sum().reset_index().rename({'Datetime': timeframe}, axis=1)
    dff = dff.drop(columns=[
        col for col in dff.columns if ('cyclist' in col.lower() or 'pedestrian' in col.lower() or 'motorist' in col.lower())
    ])

    dff = pd.melt(dff, id_vars=timeframe).rename(columns={'variable': 'Metric', 'value': 'Total'})
    def time_names(x):
        if timeframe == 'Month':
            return calendar.month_abbr[x]
        elif timeframe == 'Day of week':
            return calendar.day_abbr[x]
        elif timeframe == 'Hour':
            return str(x)+'h'
        else:
            return x
    dff[timeframe] = dff[timeframe].map(time_names)
        
    fig = px.line(
        dff, 
        x=timeframe, 
        y='Total', 
        color = 'Metric', 
        markers=True, 
        line_shape='spline', 
        render_mode='svg', 
        color_discrete_sequence=list(colors.values()),
        title='Evolution in NYC' if sel_region is None else f'Evolution in {neighborhood_name}'
    )
    fig.update_traces(line=dict(width=3), marker=dict(size=10))
    fig.update_layout(
        autosize=True,
        legend=dict(orientation="h"),
        margin={'l':0, 'r':0, 'b':0, 't':30, 'pad':0},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color="#7fafdf",
        legend_font_color= '#7fafdf',
        hovermode="x unified",
    )
    fig.update_xaxes(title='', visible=True, showticklabels=True, color = '#7fafdf', gridcolor = '#48556e')
    fig.update_yaxes(title='', visible=True, showticklabels=True, color = '#7fafdf', gridcolor = '#48556e')
    
    dff = df[df['Datetime'].dt.year.isin(years_list)]
    
    pergentage = dff.groupby('District')[list(colors.keys())].sum().reset_index()
    for col in colors.keys():
        pergentage[col] = pergentage[col] / pergentage[col].sum()
    pergentage = pd.melt(pergentage, id_vars=['District']).rename(columns={'variable': 'Metric', 'value': 'Percentage'})
    if sel_region is not None:
        neighborhood_name = sel_region['points'][0]['customdata'][1]
        pergentage = pergentage[pergentage['District'] == neighborhood_name]

    fig_per = px.bar(pergentage, x='Metric', y='Percentage', color = 'Metric', color_discrete_sequence=list(colors.values()))
    fig_per.update_layout(
        margin={'l':0, 'r':0, 'b':0, 't':30, 'pad':0},
        showlegend = False,
        coloraxis_showscale=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title='Percentage over NYC total' if sel_region is None else f'Percentage in {neighborhood_name} over NYC total',
        font_color="#7fafdf",
        legend_font_color= '#7fafdf',
        #legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="left", x=0),
    )
    fig_per.update_xaxes(title='', visible=True, showticklabels=True, color = '#7fafdf', gridcolor = '#48556e')
    fig_per.update_yaxes(title='', visible=True, showticklabels=True, color = '#7fafdf', showgrid=False, tickformat=".2%")
    
    return fig, fig_per
    
@callback(
    Output('treemap', 'figure'),
    [Input('slider-years', 'value'),
    Input('metric-selector', 'value')]
)
def update_treemap(years, stat_name):
    years_list = list(range(years[0], years[1]+1))
    df_filtered = df[df['Datetime'].dt.year.isin(years_list)]
    dff = df_filtered.groupby(['Borough', 'District'])[agg_cols].sum()
    dff = dff.astype(int).reset_index().rename(columns={'District': 'Neighborhood'})
    dff.columns = [col.capitalize() for col in dff.columns]
    dff['all'] = 'NYC'

    fig = px.treemap(
        dff, 
        path=['all', 'Borough', 'Neighborhood'], 
        values=stat_name, 
        color_discrete_sequence=borough_colors
    )
    fig.update_layout(
        autosize=True,
        margin={'l':0, 'r':0, 'b':0, 't':0, 'pad':0},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend_font_color= '#7fafdf',
    )
    fig.update_traces(root_color="lightgrey")
    fig.update_xaxes(title='', visible=True, showticklabels=True, color = '#7fafdf')
    fig.update_yaxes(title='', visible=True, showticklabels=True, color = '#7fafdf')
    return fig


@callback(
    Output('heatmap', 'figure'),
    [ Input('slider-years', 'value'),
     Input('metric-selector', 'value')]
)
def create_heatmap(years, metric):
    years_list = list(range(years[0], years[1]+1))
    df_filtered = df[df['Datetime'].dt.year.isin(years_list)]
    dff = df_filtered[['Datetime', metric]]
    
    by_month_and_dow = dff.groupby([dff['Datetime'].dt.month, dff['Datetime'].dt.day_of_week]).size()
    by_month_and_dow.index.names = ['Month', 'Day of week']
    by_month_and_dow = by_month_and_dow.to_frame().reset_index().rename(columns={0: metric})
    by_month_and_dow = by_month_and_dow.pivot(columns='Month', index='Day of week', values=metric)
    
    fig = px.imshow(
        by_month_and_dow, 
        y = list(calendar.day_abbr),
        x = list(calendar.month_abbr)[1:],
        color_continuous_scale='Darkmint',
        labels=dict(x="Month", y="Day of week", color=metric),
        aspect="auto",
        )
    fig.update_xaxes(side="top")
    fig.update_layout(
        autosize=True,
        margin={'l':0, 'r':0, 'b':0, 't':0, 'pad':0},
        showlegend = False,
        coloraxis_showscale=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    fig.update_xaxes(title='', visible=True, showticklabels=True, color = '#7fafdf')
    fig.update_yaxes(title='', visible=True, showticklabels=True, color = '#7fafdf')
    return fig


@callback(
    Output('evol_graph', 'figure'),
    [Input('timeframe-selector', 'value'), 
     Input('my-toggle-switch', 'value'), 
     Input('slider-years', 'value'),
     Input('metric-selector', 'value')]
)
def update_evolution_graph(timeframe, by_borough, years, metric):
    years_list = list(range(years[0], years[1]+1))
    df_filtered = df[df['Datetime'].dt.year.isin(years_list)]
    
    if timeframe == 'Year':
        filter_tf = df_filtered['Datetime'].dt.year
    elif timeframe == 'Month':
        filter_tf = df_filtered['Datetime'].dt.month
    elif timeframe == 'Day of week':
        filter_tf = df_filtered['Datetime'].dt.day_of_week
    else:
        filter_tf = df_filtered['Datetime'].dt.hour

    dff = df_filtered.groupby([filter_tf, 'Borough'])[agg_cols].sum().reset_index().rename({'Datetime': timeframe}, axis=1)
    dff = dff[['Borough', timeframe, metric]]
    if not by_borough:
        dff = dff.groupby(timeframe).sum().reset_index().drop(columns=['Borough'])
        
    def time_names(x):
        if timeframe == 'Month':
            return calendar.month_abbr[x]
        elif timeframe == 'Day of week':
            return calendar.day_abbr[x]
        elif timeframe == 'Hour':
            return str(x)+'h'
        else:
            return x
        
    dff[timeframe] = dff[timeframe].map(time_names)
    
    fig = px.line(
        dff, 
        x=timeframe, 
        y=metric, 
        color = 'Borough' if by_borough else None, 
        markers=True, 
        line_shape='spline', 
        render_mode='svg', 
        color_discrete_sequence=borough_colors
    )
    fig.update_traces(line=dict(width=3), marker=dict(size=10))
    fig.update_layout(
        autosize=True,
        margin={'l':0, 'r':0, 'b':0, 't':0, 'pad':0},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend_font_color= '#7fafdf',
    )
    fig.update_xaxes(title='', visible=True, showticklabels=True, color = '#7fafdf', gridcolor = '#48556e')
    fig.update_yaxes(title='', visible=True, showticklabels=True, color = '#7fafdf', gridcolor = '#48556e')
    return fig


@callback(
    [Output('factors_graph', 'figure'),
    Output('vehices_graph', 'figure')],
    Input('borough-selector', 'value'),
)
def update_factors_vehices_graph(boroughs):
    dff = df[df['Borough'].isin(boroughs)]
    
    factors = [facotor for factor_list in dff['Factors'] for facotor in factor_list]
    counter_factors = Counter(factors)
    counter_factors = pd.DataFrame.from_dict(counter_factors, orient='index').reset_index().rename(columns={'index': 'Factor', 0: 'Number of occurences'})
    counter_factors.sort_values(by='Number of occurences', ascending=False, inplace=True)
    fig_factors = px.bar(counter_factors.head(10), y='Factor', x='Number of occurences')

    vehicle_types = [facotor for factor_list in dff['Vehicle types'] for facotor in factor_list]
    counter_vehicle_types = Counter(vehicle_types)
    counter_vehicle_types = pd.DataFrame.from_dict(
        counter_vehicle_types, orient='index').reset_index().rename(columns={'index': 'Vehicle types', 0: 'Number of occurences'})
    counter_vehicle_types.sort_values(by='Number of occurences', ascending=False, inplace=True)
    fig_vehices = px.bar(counter_vehicle_types.head(10), y='Vehicle types', x='Number of occurences')
    
    for fig in [fig_factors, fig_vehices]:
        fig.update_layout(
            margin={'l':0, 'r':0, 'b':0, 't':10, 'pad':0},
            showlegend = False,
            coloraxis_showscale=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            #legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="left", x=0),
        )
        fig.update_traces(marker_color='#15b3a5')
        fig.update_xaxes(title='', visible=True, showticklabels=True, color = '#7fafdf', gridcolor = '#48556e')
        fig.update_yaxes(title='', visible=True, showticklabels=True, color = '#7fafdf', showgrid=False, autorange="reversed")
    return fig_factors, fig_vehices


if __name__ == '__main__':
    app.run_server(debug=True)

