import pandas as pd
import joblib
import numpy as np

vectorizer = joblib.load('Scripts//intent//vectorizer.sav')

def getintent(MyText):
    testdata=pd.read_csv('Scripts//intent//x_test1.csv').squeeze()
    testdata.loc[len(testdata)]=MyText

    testdata = vectorizer.transform(testdata)
    model=joblib.load('Scripts//intent//ensemble.sav')

    threshold=0.50
    y_prob =model.predict_proba(testdata)[-1]

    y_pred = np.array([np.argmax(prob) if max(prob) >= threshold else -1 for prob in y_prob.reshape(1,-1)])

    label_map = {
        -1:'others',
    0: 'thanking',
    1: 'acknowledging',
    2: 'agreeing',
    3: 'encouraging',
    4: 'questioning',
    5: 'suggesting',
    6: 'sympathizing',
    7: 'greeting',
    8: 'wishing'
    }
    y_pred=y_pred.tolist()
    for key, value in label_map.items():
        if y_pred[0] == key:
            intent = value
            break
    return str(intent)

# getintent("str")