import streamlit as st
import requests
from io import BytesIO
from PIL import Image
import base64
import json
import boto3
import numpy as np
import os

WHISPER_ENDPOINT_NAME="<<replace with respective endpoint name >>"
LLAMA_ENDPOINT_NAME="<<replace with respective endpoint name >>"
STABLE_DIFFUSION_ENDPOINT_NAME="<<replace with respective endpoint name >>"



LyricsTxt=""
LyricsImagePrompt=""
catchPrompt=""

# Generate Text based on Type
def generateTextFromLLAMA(contextTxt, promptType):
    if promptType == 'IMAGE_PROMPT':
        prompt = "Write a stunning image prompt for "+ "".join(contextTxt)
        payload = {
            "inputs":[[
                {"role": "system", "content": "You are an artist who is good at image prompt"},
                 {"role": "user", "content": prompt}
                       ]],
            "parameters":{"max_new_tokens" :256,"top_p": 0.9,"temperature": 0.6}
        }
    else:
        prompt = "Write a tempting question on this song(Use emoji when possible). Song is" +contextTxt
        payload = {
            "inputs":[[
                {"role": "system", "content": "You are an Quiz Master. who is good at asking tempting question"},
                 {"role": "user", "content": prompt}
                       ]],
            "parameters":{"max_new_tokens" :256,"top_p": 0.9,"temperature": 0.6}
        }
    
    client = boto3.client('sagemaker-runtime')
    content_type = "application/json"
    custom_attributes="accept_eula=true"
    encoded_text = json.dumps(payload).encode('utf-8')

    ## Call LLAMA2
    response = client.invoke_endpoint(EndpointName=LLAMA_ENDPOINT_NAME,
                                       CustomAttributes=custom_attributes, 
                                       ContentType=content_type, 
                                       Body=encoded_text)
    
    model_predictions =json.loads(response['Body'].read())
    model_predictions_txt=model_predictions[0]
    model_predictions_txtL1=model_predictions_txt['generation']
    generated_text=(model_predictions_txtL1['content'])
    return generated_text


#generate image from text.
def generateImageFromSD(text, col1):
    data = {"text": text}
    client = boto3.client('sagemaker-runtime')
    content_type = "application/x-text"
    encoded_text = json.dumps(data).encode('utf-8')
    #call stable siffusion
    response = client.invoke_endpoint(EndpointName=STABLE_DIFFUSION_ENDPOINT_NAME,
                                        ContentType=content_type, 
                                        Body=encoded_text)
    response_payload = json.loads(response['Body'].read().decode("utf-8"))                                             
    col1.image(np.array(response_payload["generated_image"]))
    return 


#Function to upLoad audio file
def upload_audio():
    lyricsTxt=""
    lyricsImagePrompt=""


    uploaded_file = st.file_uploader("Choose an audio file", type=['mp3','wav','mp4'])
    user_file_path=""
    if uploaded_file is not None:
        user_file_path = os.path.join("audio", uploaded_file.name)
        with open(user_file_path, "wb") as user_file:
            user_file.write(uploaded_file.getbuffer())
        
        input_audio_file_name = "./"+user_file_path
        with open(input_audio_file_name, "rb") as file:
            wav_file_read = file.read()

        #whisper model
        client = boto3.client('sagemaker-runtime')
        response = client.invoke_endpoint(EndpointName=WHISPER_ENDPOINT_NAME, 
                                          ContentType='audio/wav', 
                                          Body=wav_file_read)
        model_predictions = json.loads(response['Body'].read())
        lyricsTxt = model_predictions ['text']
        lyricsImagePrompt=generateTextFromLLAMA(lyricsTxt, "IMAGE_PROMPT")
        
    return uploaded_file, lyricsTxt,lyricsImagePrompt

# Function to fetch an image from an API
def fetch_image(myLyric,col1):
    generateImageFromSD(myLyric,col1)    
    return

 # Function to fetch text from an API
def fetch_text(lyricsTxt):
    catchPrompt=generateTextFromLLAMA(lyricsTxt,"SONG_DESC" )
    return catchPrompt   


# Function to display the uploaded audio file with a play button
def displayAudio(file):
    audio_bytes = file.read()
    st.audio(audio_bytes, format='audio/' + file.type.split('/')[-1])

# Main part of the Streamlit app
st.title("Model showcase : Whisper, Stable diffusion and Llama 2")
file, LyricsTxt,LyricsImagePrompt = upload_audio()
myLyric = "".join(LyricsTxt)

if file:
    col1, col2 = st.columns(2)
    with col1:
        grImg=fetch_image(LyricsImagePrompt,col1)
        col1.write(myLyric)
    with col2:
        st.write(fetch_text(myLyric))
        displayAudio(file)
