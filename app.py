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
from youtubesearchpython import VideosSearch
import IPython.display as ipd
import sounddevice as sd
import pydub
from pydub import AudioSegment
# from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor,Wav2Vec2Processor,Wav2Vec2ForCTC


# st.write("model loading...")
# model = Wav2Vec2ForCTC.from_pretrained('music-wav2vec2-th-finetune')
# tokenizer = Wav2Vec2CTCTokenizer.from_pretrained('music-wav2vec2-th-finetune')
# st.write("Finish")

API_TOKEN = "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"
API_URL_Non = "https://api-inference.huggingface.co/models/Nonnyss/music-wav2vec2-th-finetune"
headers = {"Authorization": f"Bearer {API_TOKEN}"}

st.title('Music Recognition🎤')

def query(API_URL,data):
    response = requests.request("POST", API_URL, headers=headers, data=data)
    return json.loads(response.content.decode("utf-8"))



if st.button('Say hello'):







    samplerate = 16000  
    duration = 5 # seconds
    filename = 'quran.wav'

    st.write("start")
    mydata = sd.rec(int(samplerate * duration), samplerate=samplerate,
        channels=1, blocking=True)
    st.write("end")

    sd.wait()
    sf.write(filename, mydata, samplerate)
    #reading the voice commands
    samples, sample_rate = librosa.load(filename , sr = 16000)
    samples = librosa.resample(samples, sample_rate, 16000)
    ipd.Audio(samples,rate=16000)



    





    #files
    src = "quran.wav"
    dst = "sing.mp3"
    #convert wav to mp3
    sound = AudioSegment.from_mp3(src)
    sound.export(dst, format="wav")




    audio_file = open(dst, 'rb')
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format='audio/mp3')
    



    if dst!=0:
        output = query(API_URL_Non,dst)
        
        with st.expander("Result"):
            predict = output.get('text')

                
            videosSearch = VideosSearch(predict, limit = 2)
            result = videosSearch.result()
            l = result.get('result')
            st.subheader(l[0].get('title'))

            watch = l[0].get('link')

            frame = f'<iframe src="http://www.youtube.com/embed/{watch[-11:]}" width="560" height="315" frameborder="0" allowfullscreen></iframe>'
            st.markdown(frame, unsafe_allow_html=True)

    else:

        paths = glob('/content/split_*.wav')
        sentence = ""
        for i in range(len(paths)):
            path = f'/content/split_{i+1}.wav'
            with open(f"{path}", "rb") as wavfile:
                input_wav = wavfile.read()
                rate, data = read(io.BytesIO(input_wav))
                bytes_wav = bytes()
                byte_io = io.BytesIO(bytes_wav)
            write(byte_io, rate, data)
            audio = byte_io.read()
                
            while(True):
                Te = query(API_URL_Non,audio)
                if 'text' in Te:
                    sentence1+=Te['text']
                    break


        with st.expander("Result"):
            st.write(f"{sentence}")







