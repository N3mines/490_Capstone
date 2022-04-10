import dash
import dash_table
import dash_uploader as du
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State, MATCH

import os
import PIL
import base64
import numpy as np
import matplotlib.pyplot as plt

import packages.tensor as td

from io import BytesIO
from PIL import Image
from pathlib import Path
from zipfile import ZipFile

from packages.targeted_callbacks import targeted_callback

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(name=__name__, external_stylesheets=external_stylesheets)

server = app.server

# This will change from what computer is running the program, in this case it
# is made for a Mac
UPLOAD_FOLDER = 'dash_data'
du.configure_upload(app, UPLOAD_FOLDER)


tensor = {} 
# Initializing these variables here.  Could be changed to allow users control
# of the values
batch_size = 3
img_height = 180
img_width = 180


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
            html.H2('Image Trainer', className='display-4'),
            html.Hr(),
            du.Upload(
                text='Select Images to Classify',
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

def parse_data(dir_name):
    classes = []
    f = os.path.join(tensor[dir_name].session_dir, dir_name)
    for file_name in os.listdir(f): 
        f2 = os.path.join(f, file_name)
        if os.path.isdir(f2):
            classes.append(
                    html.Li(
                        id=f2,
                        children=file_name
                    ))

    card_data = [
            html.Div(
                children=[
                    html.H3(
                        id={'type': 'dir_path', 'index': dir_name},
                        children=dir_name
                    ),
                    html.H5(
                        id={'type': 'classes', 'index': dir_name},
                        children='Directory Classes Found:'
                    ),
                    html.Ul(
                        id={'type': 'file_classes', 'index': dir_name},
                        children=classes
                    ),
                    html.Button(
                        id={'type': 'train_model', 'index': dir_name},
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
                        id={'type': 'training', 'index': dir_name},
                        children=''
                    ),
                    html.H5(
                        id={'type': 'trained', 'index': dir_name},
                        children=''
                    ),
                    html.H5(
                        id={'type': 'prediction_result', 'index': dir_name},
                        children=''
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

def apply_fields(file_name):
    fields = [
            dcc.Upload(
                id={'type': 'upload_images', 'index': file_name},
                children=html.Div([
                    html.Button('Upload Image to Classify')]
                )
            )]

    return html.Div(
            children = fields
            )


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


        for file_name in file_names:
            # zipfile sandard library for unzipping the images?
            with ZipFile(root_folder+'/'+file_name, 'r') as zip:
                zip.extractall('./'+root_folder)
                
        for file_name in os.listdir(root_folder):
            f = os.path.join(root_folder, file_name)
            if os.path.isdir(f) and file_name != '__MACOSX':
                tensor[file_name] = td.tensor_data()
                tensor[file_name].batch_size = batch_size
                tensor[file_name].img_height = img_height 
                tensor[file_name].img_width = img_width 

                tensor[file_name].session_dir = root_folder
                children.append(parse_data(file_name))

    return children


def train_feedback(children):
    global tensor

    print(tensor)
    
    input_states = dash.callback_context.states
    state_iter = iter(input_states.values())
    file_name = next(state_iter)

    f = os.path.join(tensor[file_name].session_dir, file_name)
    tensor[file_name].build_tensor(f)

    tensor[file_name].train_tensor()

    children = []
    children.append('Model Trained!')
    children.append(apply_fields(file_name))

    return children

targeted_callback(
        train_feedback,
        Input({'type': 'train_model', 'index': MATCH}, 'n_clicks'),
        Output({'type': 'trained', 'index': MATCH}, 'children'),
        State({'type': 'dir_path', 'index': MATCH}, 'children'),
        app=app)

def pretrain_feedback(n_clicks):
    return 'Training! Please wait :)'

targeted_callback(
        pretrain_feedback,
        Input({'type': 'train_model', 'index': MATCH}, 'n_clicks'),
        Output({'type': 'training', 'index': MATCH}, 'children'),
        app=app)

@app.callback(
        Output({'type': 'prediction_result', 'index': MATCH}, 'children'),
        Input({'type': 'upload_images', 'index': MATCH}, 'contents'),
        prevent_initial_call = True)
def update_output(file):
    if file is None:
        raise dash.exceptions.PreventUpdate
    file = file.split(',')
    image = Image.open(BytesIO(base64.b64decode(file[1]))) 
    rgb = Image.new('RGB', image.size)
    rgb.paste(image)
    image = rgb
    image = image.resize((img_height,img_width))
    np_image = np.array(image)
    
    print(np_image)
        

    return []


if __name__ == '__main__':
    app.run_server(debug=True)
