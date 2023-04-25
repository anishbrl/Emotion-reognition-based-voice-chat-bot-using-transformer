from keras.models import load_model
import tensorflow as tf
import speech_recognition as sr
from Scripts.get_emotion import *
from Scripts.generator import *
from Scripts.intent import *
from Scripts.history import *
import pandas as pd
import random
import requests
import chardet

def ask_mood():
    ask="What kind of song would you like to listen to? Say 'pop' for Pop songs, 'groovy' for Dance songs, 'zen' for Soothing songs, 'emotics' for Emote Mix, 'romantic' for Romantic songs."
    return ask

def get_song(genre):
    with open('songs_list.csv', 'rb') as f:
        result = chardet.detect(f.read())
    songs = pd.read_csv('songs_list.csv', encoding=result['encoding'])
    print(songs.info())

    exists=False
    list_songs = []
    for i in range(len(songs)):
        if genre in songs['Emotion'][i]:
            exists=True
            list_songs.append(songs.loc[i,['Name', 'Url']])
    if exists:
        chosen=random.choice(list_songs)
        song_name=str(chosen[0])
        song_url=str(chosen[1])
        song="I suggest you listen to "+"'"+song_name+"'. "+" URL: "+song_url
    else:
        song = "Oops! Sorry, I dont have any song to suggest right now. "+" (˶ᵔᵕᵔ˶) "
    return song

def getwrite(sentence):
    joke_keywords = ['tell me a joke', 'can you tell me a joke', 'do you know a joke', 'will you tell me a joke',
                     "share a joke with me","entertain me with a joke","i could use a laugh","whats a good joke you know",
                    "jokes please","make me laugh","im in the mood for a joke","do you have any jokes up your sleeve","joke time","give me a joke",
                      'i need jokes',"can I hear a joke","i need a joke stat","you are funny tell me more",'i need joke',"tell me another joke"]
    
    joke_requested = False
    response=''
    for keyword in joke_keywords:
        if keyword in sentence:
            joke_requested = True
            # Retrieve the joke from the API
            url = "https://v2.jokeapi.dev/joke/Any?blacklistFlags=nsfw,religious,political,racist,sexist,explicit&type=single"
            response = requests.get(url)
            break
    
    music_keywords=["suggest me a song",'tell me a good song',"can you recommend a song for me",'tell me a song',
                    "what's a good song you think I should listen to",
                    "i am looking for some new music any suggestions",
                    "do you have a song you can recommend to me"]
    
    music_requested=False
    for keyword in music_keywords:
        if keyword in sentence:
            music_requested=True
            break
    
    genre_keywords=['pop','zen','romantic','emotics','groovy']
    genre_requested=False
    for keyword in genre_keywords:
        if keyword in sentence:
            genre_requested=True
            break
    return music_requested,joke_requested,genre_requested,response


def predict(test,sentence,emotion,intent,history,tokenizer, START_TOKEN,END_TOKEN,VOCAB_SIZE):
    prediction = evaluate(test,sentence,emotion,intent,history,tokenizer, START_TOKEN,END_TOKEN,VOCAB_SIZE)
    predicted_sentence = tokenizer.decode(
        [i for i in prediction if i < tokenizer.vocab_size]
    )    
    return predicted_sentence


def get_joke_music(response):
    if response.status_code == 200:
        data = response.json()
        predicted_sentence = str("Sure!! Joke: "+data['joke']+" LOL 😂. "+ask_mood())
    else:
        predicted_sentence="Oops! Sorry, maybe next time. "+" (˶ᵔᵕᵔ˶) "
    return predicted_sentence

def get_joke(response):
    if response.status_code == 200:
        data = response.json()
        predicted_sentence = str("Sure!!" +data['joke']+" 😂 ")
    else:
        predicted_sentence="Oops! Sorry, maybe next time. "+" (˶ᵔᵕᵔ˶) "
    return predicted_sentence


def generater(MyText,tokenizer, START_TOKEN,END_TOKEN,VOCAB_SIZE,audiofile):
    if MyText=="...":
        reply='Are you trying to say something? I did not catch that. Could you please repeat?'
    else:
        audio=audiofile
        MyText=MyText.lower()
        #GET if the user wants jokes or music or simple reply
        music_requested,joke_requested,genre_requested,response=getwrite(MyText)
        if joke_requested and music_requested:
            reply=get_joke_music(response)
        else:
            if joke_requested:
                reply=get_joke(response)
            elif music_requested:
                reply= ask_mood()
            elif genre_requested:
                reply= get_song(MyText)
            else:
                #GETTING EMOTIION
                emotion=getemotion(audio,MyText) # gets emotion from text and speech get_emotion.py
                print ("MyText--------",MyText)
                print('EMOTION-------',emotion)
                #GETTING HISTORY
                history=gethistory()
                print("HISTORY-------",history)
                #GETTING INTENT
                intent=getintent(MyText)
                print('INTENT-------',intent)
                #GETTING REPLY
                test=create_model(VOCAB_SIZE)
                test.load_weights('Scripts\model_checkpoints\weights-improvement-31.h5')
                reply=predict(test,MyText,emotion,intent,history,tokenizer, START_TOKEN,END_TOKEN,VOCAB_SIZE)
        print("REPLY-----------",reply)
        sethistory(MyText,reply,music_requested,joke_requested,genre_requested)
    return reply
    


def get_mytext(audiofile):
    r = sr.Recognizer() 
    hellow=sr.AudioFile(audiofile)
    with hellow as source:
        audio = r.record(source)
    try:
        MyText2 = r.recognize_google(audio)
        print (MyText2)
    except sr.RequestError as e:
        MyText2="..."
    except sr.UnknownValueError:
        MyText2="..."       
    return MyText2


