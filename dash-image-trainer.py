import dash
import dash_uploader as du
from dash import dcc, html, dash_table
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
import zipfile

# Code in targeted_callback is credited with github link in file
from packages.targeted_callbacks import targeted_callback

# CSS Styling for the website
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
batch_size = 32
img_height = 180
img_width = 180

# Styling for the sidebar on website
SIDEBAR_STYLE = {
        'position': 'fixed',
        'top': 0,
        'left': 0,
        'bottom': 0,
        'padding': '2rem 1rem',
        'background-color': '#f8f9fa',
        'width': '15%'
    }

# Styling for the main content on the website
CONTENT_STYLE = {
        'margin-left': '17%',
        'padding': '2rem 1rem'
    }

# Setting up the sidebar with buttons and labels the user can use
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

# Setting up blank content since it will be populated by uploaded data later
content = html.Div([
            html.Div(id='output_image_upload')
        ],
        style=CONTENT_STYLE
    )

# Populating cards in the content on the page.  Data from the uploaded zip file
# is processed and classes pulled.  Most of the card will still be blank after
# this step but will change as the user interacts and does more
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
                    html.H5('Select epochs for training'),
                    dcc.Slider(
                        id={'type': 'epochs', 'index': dir_name},
                        min=0,
                        max=10,
                        step=1,
                        value=5,
                        tooltip={'placement': 'bottom'}
                    ),
                    dcc.Checklist(
                        id={'type': 'randomize', 'index': dir_name},
                        options=[{'label': 'Randomize Image Data', 'value': 'RI'}],
                        value=['RI']
                    ),
                    html.Button(
                        id={'type': 'train_model', 'index': dir_name},
                        children='Train on Data',
                        style={'padding-bottom':'3rem'},
                    )
                ],
                style={
                    'width': '40%',
                    'padding': '3rem 3rem 3rem 3rem',
                    'float': 'left'}
            ),
            # This is the section that will get populated by the user later on
            # each with unique IDs to reference
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
                    ),
                    html.Div(
                        id={'type': 'prediction_img', 'index': dir_name},
                    )
                ],
                style={
                    'width': '40%',
                    'padding': '3rem 3rem 10rem 3rem',
                    'float': 'left'
                }
            )]
    # Styling for the card containing the content.  Some thought was put into
    # it so that you can tell the difference between two different datasets
    # based off of it's look
    return html.Div(
            children=card_data,
            style={
                'box-shadow': '0 4px 8px 0 rgba(0,0,0,0.2)',
                'border-radius': '20px',
                'padding': '30px 30px',
                'margin': '10px',
                'overflow': 'auto'
            })

# Populating the upload data section with the same unique ID pattern
def apply_fields(file_name):
    fields = [
            dcc.Upload(
                id={'type': 'upload_images', 'index': file_name},
                children=html.Div([
                    html.Button('Upload Image to Classify')]
                )
            ),
            html.Button('Download Model', 
                id={'type': 'download_button', 'index': file_name}),
            dcc.Download(id={'type': 'download_model', 'index': file_name})
        ]

    return html.Div(
            children = fields
            )


# Setting up initial webpage layout
app.layout = html.Div([sidebar, content])

### 
##
# This is all going to be callbacks and the interactions that the users
# triggers
##
###

# Callback to add cards to the website content.  One issue present here is that
# state isn't saved so if the user adds another dataset, everything is
# re-applied to the content section.  An improvement here would be to add state
# which is complicated and didn't really work.
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

        # Extract data from the uploaded zip file.
        for file_name in file_names:
            # zipfile sandard library for unzipping the images?
            with zipfile.ZipFile(root_folder+'/'+file_name, 'r') as zip:
                zip.extractall('./'+root_folder)
        
        # Go through all the directories of images and add a card to the list.
        # This will also write to the global dictionary so that data can be
        # accessed later for tensor
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

# This defenition is using the targeted_callback method which allows multiple
# functions to have the same output.  This didn't quite work with lots of
# testing, but the functionw as never changed back to the main callback layout

# This defenition is just used to start the tensor training and return a
# message when it's done
def train_feedback(children):
    global tensor

     
    input_states = dash.callback_context.states
    state_iter = iter(input_states.values())
    file_name = next(state_iter)
    epochs = next(state_iter)
    randomize = next(state_iter)

    f = os.path.join(tensor[file_name].session_dir, file_name)
    tensor[file_name].build_tensor(f, randomize)

    tensor[file_name].train_tensor(epochs)

    children = []
    children.append('Model Trained!')
    children.append(apply_fields(file_name))

    return children

# Part of the train_feedback callback
targeted_callback(
        train_feedback,
        Input({'type': 'train_model', 'index': MATCH}, 'n_clicks'),
        Output({'type': 'trained', 'index': MATCH}, 'children'),
        State({'type': 'dir_path', 'index': MATCH}, 'children'),
        State({'type': 'epochs', 'index': MATCH}, 'value'),
        State({'type': 'randomize', 'index': MATCH}, 'value'),
        app=app)

# Message for when training button is initially clicked so user knows something
# is happening
def pretrain_feedback(n_clicks):
    return 'Training! Please wait :)'

# Part of the pretrain_feedback
targeted_callback(
        pretrain_feedback,
        Input({'type': 'train_model', 'index': MATCH}, 'n_clicks'),
        Output({'type': 'training', 'index': MATCH}, 'children'),
        app=app)

# Add the prediction results and image it predicted on to the card.
@app.callback(
        Output({'type': 'prediction_result', 'index': MATCH}, 'children'),
        Output({'type': 'prediction_img', 'index': MATCH}, 'children'),
        Input({'type': 'upload_images', 'index': MATCH}, 'contents'),
        State({'type': 'dir_path', 'index': MATCH}, 'children'),
        prevent_initial_call = True)
def update_output(file, file_name):
    global tensor
    if file is None:
        raise dash.exceptions.PreventUpdate

    image_ret = [html.Img(src=file)]

    file = file.split(',')
    # Take in base64 string of image and convert it to PIL format
    image = Image.open(BytesIO(base64.b64decode(file[1]))) 
    image = image.resize((img_height, img_width))

    score = tensor[file_name].predict_tensor(image)
    
    output = (tensor[file_name].class_names[np.argmax(score)]+" with "+
        str(100*np.max(score)) + " percent confidence")
        

    return [output], image_ret


# Download model callback for user to save whatever they ended up using
@app.callback(
        Output({'type': 'download_model', 'index': MATCH}, 'data'),
        Input({'type': 'download_button', 'index': MATCH}, 'n_clicks'),
        State({'type': 'dir_path', 'index': MATCH}, 'children'),
        prevent_initial_call = True)
def download_tensor(n_clicks, file_name):
    global tensor
    f = os.path.join(tensor[file_name].session_dir, file_name)
    # Save model on serverside
    tensor[file_name].model.save(f+'/my_model.h5')

    # Send file over to the user 
    ret = dcc.send_file(f+'/my_model.h5')
   
    # Remove the file serverside 
    os.remove(f+'/my_model.h5')

    # This could all probably be done in memeory but I couldn't seem to figure
    # out how to make tensorflow work with me in the regard

    return ret


if __name__ == '__main__':
    app.run_server(debug=True)
