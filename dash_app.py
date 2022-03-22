import dash
from dash.dependencies import Input, Output, State, MATCH
import dash_core_components as dcc
import dash_html_components as html
import dash_table

from packages.targeted_callbacks import targeted_callback

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

SIDEBAR_STYLE = {
        'position': 'fixed',
        'top': 0,
        'left': 0,
        'bottom': 0,
        'padding': '2rem 1rem',
        'background-color': '#f8f9fa',
        'width': '15%'}

CONTENT_STYLE = {
        'margin-left': '17%',
        'padding': '2rem 1rem'}

sidebar = html.Div(
        id='sidebar',
        children=[
            html.H2("Image Trainer", className="display-4"),
            html.Hr(),
            html.P(
                "options for model training and other pieces",
                className="lead")],
        style=SIDEBAR_STYLE)

content = html.Div([
        html.Div(
            children='Main page where tensorflow work will be',
            style={'textAlign': 'center'}),
        html.Div(id='output_data_upload')],
    style=CONTENT_STYLE)

# Setting up initial webpage layout
app.layout = html.Div([sidebar, content])

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0')
