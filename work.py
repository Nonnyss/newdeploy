import streamlit as st
import requests
import json
import os
import librosa, librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np
import IPython
from scipy.io import wavfile
import scipy.signal
import time
from datetime import timedelta as td
from glob import glob
from scipy.io.wavfile import read, write
import io
import sounddevice as sd
import IPython.display as ipd
from youtubesearchpython import VideosSearch
#import streamlit.components.v1 as components
from pydub import AudioSegment
import pydub

API_TOKEN = "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"
API_URL_Non = "https://api-inference.huggingface.co/models/Nonnyss/music-wav2vec2-th-finetune"
headers = {"Authorization": f"Bearer {API_TOKEN}"}

#parent_dir = os.path.dirname(os.path.abspath(__file__))
#build_dir = os.path.join(parent_dir, "st_audiorec/frontend/build")
#st_audiorec = components.declare_component("st_audiorec", path=build_dir)


st.title('Music Recognition🎤')

def query(API_URL,data):
    response = requests.request("POST", API_URL, headers=headers, data=data)
    return json.loads(response.content.decode("utf-8"))


def search(audio):
    output = query(API_URL_Non,audio)           
    #with st.expander("Result"):
        predict = output.get('text')
        st.write(predict)
                
        videosSearch = VideosSearch(predict, limit = 2)
        result = videosSearch.result()
        l = result.get('result')
        l = l[0].get('title')

        watch = l[0].get('link')

        frame = f'<iframe src="http://www.youtube.com/embed/{watch[-11:]}" width="560" height="315" frameborder="0" allowfullscreen></iframe>'
        st.markdown(frame, unsafe_allow_html=True)
                
                
def audio_file():
    uploaded_file = st.file_uploader("Upload file")
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        st.audio(bytes_data,'audio/mp3')
        if st.button("Find your fav Song!"):
            search(bytes_data)

def voice():
    if st.button('Say hello'): 
        samplerate = 3000  
        duration = 5 # seconds
        filename = 'quran.wav'

        st.write("start")
        mydata = sd.rec(int(samplerate * duration), samplerate=samplerate,
            channels=1, blocking=True)
        st.write("end")

        sd.wait()

        sf.write(filename, mydata, samplerate)
        #reading the voice commands
        samples, sample_rate = librosa.load(filename , sr = 3000)
        samples = librosa.resample(samples, sample_rate,3000)
        ipd.Audio(samples,rate=3000)

        #files
        src = "quran.wav"
        dst = "sing.mp3"
        #convert wav to mp3
        sound = AudioSegment.from_mp3(src)
        sound.export(dst, format="wav")
        audio_file = open(dst, 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/mp3')  
        search(dst)  




option = st.sidebar.selectbox(
            'Input',
            ('Audio 🎸 ','Sing 🎼 '))

st.write('Your selected:', option)
if option == 'Audio 🎸 ':
    audio_file()
if option == 'Sing 🎼 ':
    voice()