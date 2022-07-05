#import packages
import random
import json

import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

#import google translator
from googletrans import Translator


#translator usage
translator = Translator()


#setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#opening intents file with data in read mode
with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

#opening data.pth
FILE = "data.pth"
data = torch.load(FILE)

#getting information from file
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

#initialize bot name
bot_name = "Very smart bilingual chatbot"
print("Let's chat! (type 'quit' to exit)")


while True:
    #example sentence = "What is your hobby?"
    sentence = input("You: ")
    #typing quit interrupts the program
    if sentence == "quit":
        break

    # Detecting language
    detected_lang = translator.detect(sentence)
    detected_lang = str(detected_lang)

    detected_lang = detected_lang[detected_lang.find('lang='):detected_lang.find(',')]


    # If PL input -> translate to EN
    if detected_lang[5:] == 'pl':
        sentence = translator.translate(sentence, dest='en')

    # Google transaltor return the following type: <class 'googletrans.models.Translated'> and it can't be processed by
    # tokenizator, we have to get only translated string:
        sentence = str(sentence)
        # print(sentence)
        sentence = sentence[sentence.find('text='):sentence.find(', p')]


    #print(type(sentence))
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                # Standard bot response
                response = {random.choice(intent['responses'])}
                # print(response)
                response = repr(response)
                if detected_lang[5:] == 'pl':
                    # Answer using en -> pl translator
                    pl_resp = translator.translate(response, dest='pl')
                    pl_resp = str(pl_resp)
                    # print(pl_resp)
                    print(bot_name + ':' + pl_resp[pl_resp.find("{"):pl_resp.find(", pr")])
                else:
                    # Answer using en -> en translator
                    en_resp = translator.translate(response, dest='en')
                    en_resp = str(en_resp)
                    print(bot_name + ':' + en_resp[en_resp.find("{"):en_resp.find(", pr")])
    else:
        if detected_lang == 'en':
            print(f"{bot_name}: I do not understand...")
        else:
            print(f"{bot_name}: Nie rozumiem co masz na myśli.")
