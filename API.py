from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from Scripts.test2 import *
from Scripts.history import *
from Scripts.intent import *
from Scripts.get_emotion import *
import uuid
from fastapi.responses import HTMLResponse
import subprocess
app=FastAPI()
import tensorflow_datasets as tfds
import os

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=[""],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return{'Chatbot'}


def buildtoken():
    tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file('Scripts\intent+emotion+context.tf')
    START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]
    VOCAB_SIZE = tokenizer.vocab_size + 2
    return tokenizer, START_TOKEN,END_TOKEN,VOCAB_SIZE

tokenizer, START_TOKEN,END_TOKEN,VOCAB_SIZE=buildtoken()


path=""
text=""
@app.post('/audio')
def upload_audio(audio:UploadFile=File(...)):
    global path
    global text
    path=''
    text=''
    file_location = f"static/audio/{uuid.uuid1()}{audio.filename}"
    with open (file_location, "wb+") as file_object:
        file_object.write(audio.file.read())
    dest_path=f'static/audio/{uuid.uuid1()}test.wav'
    command = f'ffmpeg -i {file_location} {dest_path}'
    subprocess.call(command,shell=True)
    MyText=get_mytext(dest_path)
    text=MyText
    path=dest_path
    os.remove(file_location)
    return {"texts":MyText}

# @app.get('/stop')
# def stop_reply():
#     print('STOPPED')
#     return {"reply": "OKAY"}

@app.get('/reply')
def get_reply():
    MyText=''
    reply1=''
    dest_path=path
    MyText=text
    reply1=generater(MyText,tokenizer, START_TOKEN,END_TOKEN,VOCAB_SIZE,dest_path)
    print("****reply:",reply1)
    print("****mytext:",MyText)
    os.remove(dest_path)
    return {"reply": reply1}

@app.get('/clear')
def clear_history():
    print("PAGE RELOADED SUCCESS!!")
    clear1()
    return 0

