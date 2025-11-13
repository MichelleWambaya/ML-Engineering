"""
Interactive Dashboard for Literacy Rates Analysis
Creates a Dash web application for visualizing literacy data
"""

import pandas as pd
import numpy as np
from pathlib import Path
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go

class LiteracyDashboard:
    """Interactive dashboard for literacy analysis"""
    
    def __init__(self, data_path="data/processed/literacy_data_processed.csv"):
        self.data_path = Path(data_path)
        self.df = None
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.setup_layout()
        self.setup_callbacks()
    
    def load_data(self):
        """Load processed data"""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data not found at {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        return self.df
    
    def setup_layout(self):
        """Setup dashboard layout"""
        self.app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("Literacy Rates Analysis - Kenyan Counties", 
                           className="text-center mb-4"),
                    html.Hr()
                ])
            ]),
            
            dbc.Row([
                dbc.Col([
                    html.Label("Select County:"),
                    dcc.Dropdown(
                        id='county-dropdown',
                        options=[{'label': county, 'value': county} 
                                for county in sorted(self.df['county'].unique())] if self.df is not None else [],
                        value='Nairobi' if self.df is not None else None,
                        clearable=False
                    )
                ], md=4),
                dbc.Col([
                    html.Label("Select Year:"),
                    dcc.Dropdown(
                        id='year-dropdown',
                        options=[{'label': str(year), 'value': year} 
                                for year in sorted(self.df['year'].unique())] if self.df is not None else [],
                        value=max(self.df['year'].unique()) if self.df is not None else None,
                        clearable=False
                    )
                ], md=4),
                dbc.Col([
                    html.Label("Select Metric:"),
                    dcc.Dropdown(
                        id='metric-dropdown',
                        options=[
                            {'label': 'Overall Literacy Rate', 'value': 'literacy_rate'},
                            {'label': 'Male Literacy Rate', 'value': 'male_literacy_rate'},
                            {'label': 'Female Literacy Rate', 'value': 'female_literacy_rate'},
                            {'label': 'Gender Gap', 'value': 'gender_gap'}
                        ],
                        value='literacy_rate',
                        clearable=False
                    )
                ], md=4)
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='literacy-trend-chart')
                ], md=6),
                dbc.Col([
                    dcc.Graph(id='county-comparison-chart')
                ], md=6)
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='gender-comparison-chart')
                ], md=6),
                dbc.Col([
                    dcc.Graph(id='infrastructure-chart')
                ], md=6)
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    html.H3("Key Statistics", className="mb-3"),
                    html.Div(id='key-statistics')
                ], md=12)
            ])
        ], fluid=True)
    
    def setup_callbacks(self):
        """Setup dashboard callbacks"""
        
        @self.app.callback(
            [Output('literacy-trend-chart', 'figure'),
             Output('county-comparison-chart', 'figure'),
             Output('gender-comparison-chart', 'figure'),
             Output('infrastructure-chart', 'figure'),
             Output('key-statistics', 'children')],
            [Input('county-dropdown', 'value'),
             Input('year-dropdown', 'value'),
             Input('metric-dropdown', 'value')]
        )
        def update_charts(county, year, metric):
            # Filter data
            county_data = self.df[self.df['county'] == county] if county else self.df
            year_data = self.df[self.df['year'] == year] if year else self.df
            
            # Trend chart
            trend_fig = px.line(
                county_data,
                x='year',
                y=metric,
                title=f'{metric.replace("_", " ").title()} Trend for {county}',
                labels={metric: metric.replace("_", " ").title()}
            )
            trend_fig.update_layout(template='plotly_white')
            
            # County comparison
            top_counties = year_data.nlargest(15, metric)
            comparison_fig = px.bar(
                top_counties,
                x=metric,
                y='county',
                orientation='h',
                title=f'Top 15 Counties by {metric.replace("_", " ").title()} ({year})',
                labels={metric: metric.replace("_", " ").title()}
            )
            comparison_fig.update_layout(template='plotly_white', yaxis={'categoryorder': 'total ascending'})
            
            # Gender comparison
            gender_data = year_data[['county', 'male_literacy_rate', 'female_literacy_rate']].head(20)
            gender_fig = go.Figure()
            gender_fig.add_trace(go.Bar(
                x=gender_data['county'],
                y=gender_data['male_literacy_rate'],
                name='Male',
                marker_color='lightblue'
            ))
            gender_fig.add_trace(go.Bar(
                x=gender_data['county'],
                y=gender_data['female_literacy_rate'],
                name='Female',
                marker_color='lightcoral'
            ))
            gender_fig.update_layout(
                title='Gender Literacy Comparison (Top 20 Counties)',
                xaxis_title='County',
                yaxis_title='Literacy Rate (%)',
                template='plotly_white',
                barmode='group'
            )
            
            # Infrastructure chart
            infra_data = year_data[['county', 'schools_per_100k', 'teachers_per_100k', 
                                   'libraries_per_100k']].head(15)
            infra_fig = go.Figure()
            infra_fig.add_trace(go.Scatter(
                x=infra_data['county'],
                y=infra_data['schools_per_100k'],
                mode='lines+markers',
                name='Schools per 100k',
                line=dict(color='green')
            ))
            infra_fig.add_trace(go.Scatter(
                x=infra_data['county'],
                y=infra_data['teachers_per_100k'],
                mode='lines+markers',
                name='Teachers per 100k',
                line=dict(color='blue')
            ))
            infra_fig.update_layout(
                title='Education Infrastructure Comparison',
                xaxis_title='County',
                yaxis_title='Per 100k Population',
                template='plotly_white'
            )
            
            # Key statistics
            current_data = self.df[(self.df['county'] == county) & (self.df['year'] == year)]
            if len(current_data) > 0:
                stats = current_data.iloc[0]
                stats_html = dbc.Row([
                    dbc.Col([
                        html.H5(f"Literacy Rate: {stats['literacy_rate']:.1f}%"),
                        html.P(f"Male: {stats['male_literacy_rate']:.1f}%"),
                        html.P(f"Female: {stats['female_literacy_rate']:.1f}%")
                    ], md=3),
                    dbc.Col([
                        html.H5(f"Schools: {stats['primary_schools'] + stats['secondary_schools']}"),
                        html.P(f"Teachers: {stats['total_teachers']}"),
                        html.P(f"Student-Teacher Ratio: {stats['student_teacher_ratio']:.1f}")
                    ], md=3),
                    dbc.Col([
                        html.H5(f"Enrollment"),
                        html.P(f"Primary: {stats['primary_enrollment_rate']:.1f}%"),
                        html.P(f"Secondary: {stats['secondary_enrollment_rate']:.1f}%")
                    ], md=3),
                    dbc.Col([
                        html.H5(f"Economic Indicators"),
                        html.P(f"GDP per Capita: ${stats['gdp_per_capita']:.0f}"),
                        html.P(f"Poverty Rate: {stats['poverty_rate']:.1f}%")
                    ], md=3)
                ])
            else:
                stats_html = html.P("No data available for selected filters")
            
            return trend_fig, comparison_fig, gender_fig, infra_fig, stats_html
    
    def run(self, host='127.0.0.1', port=8050, debug=True):
        """Run the dashboard"""
        if self.df is None:
            self.load_data()
        print(f"Starting dashboard on http://{host}:{port}")
        self.app.run_server(host=host, port=port, debug=debug)

if __name__ == "__main__":
    dashboard = LiteracyDashboard()
    dashboard.run()

