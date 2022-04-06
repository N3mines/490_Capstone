import dash
import dash_table
import dash_uploader as du
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State, MATCH

import os
import PIL
import numpy as np
import matplotlib.pyplot as plt

import packages.tensor as td

from pathlib import Path
from zipfile import ZipFile
from flask import Flask

from packages.targeted_callbacks import targeted_callback

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

server = Flask(__name__)
app = dash.Dash(name='dash_app', server=server, external_stylesheets=external_stylesheets)

# This will change from what computer is running the program, in this case it
# is made for a Mac
UPLOAD_FOLDER = "dash_data"
du.configure_upload(app, UPLOAD_FOLDER)


tensor = td.tensor_data()
# Initializing these variables here.  Could be changed to allow users control
# of the values
tensor.batch_size = 3
tensor.img_height = 180
tensor.img_width = 180


SIDEBAR_STYLE = {
        'position': 'fixed',
        'top': 0,
        'left': 0,
        'bottom': 0,
        'padding': '2rem 1rem',
        'background-color': '#f8f9fa',
        'width': '15%'
    }

CONTENT_STYLE = {
        'margin-left': '17%',
        'padding': '2rem 1rem'
    }

sidebar = html.Div(
        id='sidebar',
        children=[
            html.H2("Image Trainer", className="display-4"),
            html.Hr(),
            du.Upload(
                text='Drag and Drop .zip here',
                text_completed='Completed: ',
                pause_button=False,
                cancel_button=True,
                max_file_size=1800,  # 1800 Mb
                filetypes=['zip'],
                id='upload_data',
            )
        ],
        style=SIDEBAR_STYLE
    )

content = html.Div([
            html.Div(
                children='Main page where tensorflow work will be',
                style={'textAlign': 'center'}
            ),
            html.Div(id='output_image_upload')
        ],
        style=CONTENT_STYLE
    )

def parse_data(dir_path, dir_name):
    classes = []
    for file_name in os.listdir(dir_path): 
        f = os.path.join(dir_path, file_name)
        if os.path.isdir(f):
            classes.append(
                    html.H6(
                        id=f,
                        children=file_name
                    ))

    card_data = [
            html.Div(
                children=[
                    html.H3(
                        id={'type': 'dir_path', 'index': dir_path},
                        children=dir_name
                    ),
                    html.H5(
                        id={'type': 'classes', 'index': dir_path},
                        children='Directory Classes Found:'
                    ),
                    html.Div(
                        id={'type': 'file_classes', 'index': dir_path},
                        children=classes
                    ),
                    html.Button(
                        id={'type': 'train_model', 'index': dir_path},
                        children='Train on Data',
                        style={'padding-bottom':'3rem'}
                    )
                ],
                style={
                    'width': '40%',
                    'padding': '3rem 3rem 3rem 3rem',
                    'float': 'left'}
            ),
            html.Div(
                children=[
                    html.H5(
                        id={'type': 'trained', 'index': dir_path},
                        children=""
                    )
                ],
                style={
                    'width': '40%',
                    'padding': '3rem 3rem 10rem 3rem',
                    'float': 'left'
                }
            )]

    return html.Div(
            children=card_data,
            style={
                'box-shadow': '0 4px 8px 0 rgba(0,0,0,0.2)',
                'border-radius': '20px',
                'padding': '30px 30px',
                'margin': '10px',
                'overflow': 'auto'
            })



# Setting up initial webpage layout
app.layout = html.Div([sidebar, content])

@app.callback(
        Output('output_image_upload', 'children'),
        Input('upload_data', 'isCompleted'),
        State('upload_data', 'fileNames'),
        State('upload_data', 'upload_id'))
def update_output(is_completed, file_names, upload_id):
    global tensor
    if not is_completed:
        return

    # Previous data fields will be over written, sorry
    children = []

    if file_names is not None:
        if upload_id:
            root_folder = UPLOAD_FOLDER +'/'+  upload_id
        else:
            root_folder = UPLOAD_FOLDER

        tensor.session_dir = root_folder

        for file_name in file_names:
            # zipfile sandard library for unzipping the images?
            with ZipFile(root_folder+'/'+file_name, 'r') as zip:
                zip.extractall('./'+root_folder)
                
        for file_name in os.listdir(root_folder):
            f = os.path.join(root_folder, file_name)
            if os.path.isdir(f) and file_name != '__MACOSX':
                children.append(parse_data(f, file_name))

    return children

@app.callback(
        Output({'type': 'trained', 'index': MATCH}, 'children'),
        Input({'type': 'train_model', 'index': MATCH}, 'n_clicks'),
        State({'type': 'dir_path', 'index': MATCH}, 'children'),
        prevent_initial_call=True)
def train_feedback(n_clicks, file_name):
    global tensor

    f = os.path.join(tensor.session_dir, file_name)
    tensor.build_tensor(f)

    tensor.train_tensor()
    
    

    return "trained lmaoo"



if __name__ == '__main__':
    app.run_server(debug=True, host='10.0.0.45')
