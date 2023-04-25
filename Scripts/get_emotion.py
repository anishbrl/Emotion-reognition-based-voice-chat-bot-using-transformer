import pandas as pd
import numpy as np
import librosa
from keras.models import load_model
import joblib
import pickle
from scipy import sparse
from keras.models import load_model
from API import *

#EMOTION DETECTION from speech
def pretrained():
    model = load_model('Scripts\emotion\Emotion_Model.h5')
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    Sentiments = {  0 : "female_angry",
                  11 : "male_sad",
                   1 : "female_disgust",
                   2 : "female_fear",
                   3 : "female_happy",
                   4 : "female_neutral",
                   5 : "female_sad",
                   6 : "male_angry",
                   7 : "male_disgust",
                   8 : "male_fear",
                  9: "male_happy",
                  10 :"male_neutral"
    }
    return model,Sentiments

def extract_feature(filename):
    X, sample_rate = librosa.load(filename,sr=44100,offset=0.5)
    mfcc=np.mean(librosa.feature.mfcc(y=X,sr=sample_rate,n_mfcc=40).T,axis=0)
    return mfcc

def speech_emotion(livepreds,Sentiments):
    ave={}
    for i, prob in enumerate(livepreds[0]):
        if i==4 or i==10:
            if 'neutral' not in ave:
                ave['neutral'] = []
            ave['neutral'].append(prob)
        if i==0 or i==6:
            if 'angry' not in ave:
                ave['angry'] = []
            ave['angry'].append(prob)
        if i==11 or i==5:
            if 'sad' not in ave:
                ave['sad'] = []
            ave['sad'].append(prob)
        if i==1 or i==7:
            if 'disgust' not in ave:
                ave['disgust'] = []
            ave['disgust'].append(prob)
        if i==2 or i==8:
            if 'fear' not in ave:
                ave['fear'] = []
            ave['fear'].append(prob)
        if i==3 or i==9:
            if 'happy' not in ave:
                ave['happy'] = []
            ave['happy'].append(prob)

    for key,value in ave.items():
        ave[key]=max(value)
    return ave


def text_emotion(MyText):
    test=pd.read_csv('Scripts//emotion//x_test.csv').squeeze()
    test.loc[len(test)] = MyText
    text_model=joblib.load('Scripts\emotion\Random_Forest.sav')

    #Get Probabilities
    sample_prob = text_model.predict_proba(test)[-1]
    # print(type(sample_prob))
    emotions=['sad','angry','fear','happy','neutral','disgust']
    dic={}
    for i, val in enumerate(sample_prob):
        dic[emotions[i]] = val
    # print("dic",dic)
    return dic


def getemotion(audio,MyText):
    model,Sentiments=pretrained()
    livedf2= extract_feature(audio)
    livedf2= pd.DataFrame(data=livedf2)
    # print(livedf2.shape)
    
    if len(livedf2)<40:
        l=40-len(df)
        for i in range(l):
            livedf2.loc[len(livedf2.index)] = [0]
    livedf2 = livedf2.stack().to_frame().T
    twodim= np.expand_dims(livedf2, axis=2)

    livepreds = model.predict(twodim, batch_size=32, verbose=1)

    speech_emotion_preds=speech_emotion(livepreds,Sentiments)
    text_emotion_preds=text_emotion(MyText)

    # print("**text_emotion_preds:",text_emotion_preds)
    # print("**speech_emotion_preds:",speech_emotion_preds)

    final_dic={}
    for key in speech_emotion_preds:
        for key in text_emotion_preds:
            final_dic[key] =((speech_emotion_preds[key]*0.4) + (text_emotion_preds[key]*0.6))
    
    emotion=max(final_dic,key=final_dic.get)
    return emotion
