import re
history = ''
his_list=[]
def preprocess(sentence):
    sentence = str(sentence).strip()
    sentence=re.sub(r'\[|\]', '', sentence)
    sentence = re.sub(r"([:;/\|*-+&?])", r" \1 ", sentence)
    return sentence.lower()

def sethistory(MyText, reply,music_requested,joke_requested,genre_requested):
    global history
    if music_requested or joke_requested or genre_requested:
        reply=''
    else:
        reply=preprocess(reply)
        if len(history)>1000:
            history=str(his_list[-4:])
        else:
            history += str(MyText) + "<EOT>" + str(reply) + "<EOT>"
            his_list.append(str(MyText) + "<EOT>" + str(reply) + "<EOT>")
    return 0

def gethistory():
    global history
    if history == '':
        return "<SOC>"
    else:
        return history

def clear1():
    global history
    history=''
    print("HISTORY PAGE")
    return history