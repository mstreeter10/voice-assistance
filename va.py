# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 13:33:51 2023

@author: nokil
"""
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import reuters
from gtts import gTTS
from playsound import playsound
from datetime import datetime

import pywhatkit
import numpy as np
import os
import speech_recognition as sr
import webbrowser

word_index = reuters.get_word_index()
word_index["subtract"] = 30979
word_index["subtraction"] = 30980
word_index["multiplication"] = 30981
word_index["song"] = 30982
word_index["website"] = 30982

dimension = len(word_index)

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        for j in sequence:
            results[i, j] = 1.
    return results

def get_nums(textIn):
    results = []
    for c in textIn:
        if c.isdigit():
            results.append(c)
    return results

train_temp = [["what's", "the", "temperature"],
             ["what's", "the", "temp"],
             ["how", "humid", "is", "it"],
             ["how", "hot", "is", "it"],
             ["how", "cold", "is", "it"],
             ["how", "cool", "is", "it"],
             ["how", "warm", "is", "it"],
             ["what's", "the", "humidity"],
             ["what's", "the", "heat"],
             ["what's", "the", "water", "air", "contents"],
             ["is", "it", "cold"],
             ["is", "it", "warm"],
             ["humidity"], 
             ["temperature"],
             ["the", "hot"],
             ["the", "cold"],
             ["the", "humid"],
             ["temp"],
             ["water"],
             ["warm"],
             ["tell", "me", "the", "temperature"],
             ["tell", "me", "the", "humidity"],
             ["what's", "1", "plus", "1"],
             ["what's", "2", "plus", "2"],
             ["what's", "3", "plus", "3"],
             ["add", "1", "and", "2"],
             ["add", "2", "and", "3"],
             ["add", "3", "and", "4"],
             ["what's", "1", "minus", "1"],
             ["what's", "2", "minus", "2"],
             ["what's", "3", "minus", "3"],
             ["subtract", "3", "from", "2"],
             ["subtract", "4", "from", "3"],
             ["subtract", "5", "from", "4"],
             ["what's", "1", "times", "1"],
             ["what's", "2", "times", "2"],
             ["what's", "3", "times", "3"],
             ["multiply", "3", "and", "3"],
             ["multiply", "4", "and", "4"],
             ["multiply", "5", "and", "5"],
             ["what's", "6", "divided", "by", "3"],
             ["what's", "4", "divided", "by", "2"],
             ["what's", "2", "divided", "by", "1"],
             ["divide", "3", "by", "3"],
             ["divide", "4", "by", "4"],
             ["divide", "5", "by", "5"],
             ["can", "you", "do", "addition"],
             ["can", "you", "do", "subtraction"],
             ["can", "you", "do", "multiplication"],
             ["can", "you", "do", "division"],
             ["who", "are", "you"],
             ["what", "can", "you", "do"],
             ["what", "is", "your", "purpose"],
             ["what's", "the", "time"],
             ["tell", "me", "the", "time"],
             ["what", "time", "is", "it"],
             ["play", "music", "by"],
             ["play", "music", "on"],
             ["play"],
             ["play", "me", "song"],
             ["website"],
             ["pull", "up", "your", "website"],
             ["what's", "your", "website"]]

test_temp = [["the", "hot"],
             ["the", "cold"],
             ["the", "humid"],
             ["temp"],
             ["water"],
             ["warm"],
             ["humidity", "temperature"],
             ["tell", "me", "the", "temperature"],
             ["tell", "me", "the", "humidity"]]

train_temp_labels = [0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 2,
                     2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5,
                     2, 3, 4, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9]
test_temp_labels = [0, 0, 1, 0, 1, 0, 0, 0, 1]

train_temp_nums = train_temp.copy()
test_temp_nums = test_temp.copy()

for i, sequence in enumerate(train_temp):
    for j in range(0, len(sequence)):
        train_temp_nums[i][j] = word_index[train_temp[i][j]]
    
for i, sequence in enumerate(test_temp):
    for j in range(0, len(sequence)):
        test_temp_nums[i][j] = word_index[test_temp[i][j]]

x_train = vectorize_sequences(train_temp_nums, dimension)
x_test = vectorize_sequences(test_temp_nums, dimension)

y_train = np.array(train_temp_labels)
y_test = np.array(test_temp_labels)

model = keras.Sequential([
    layers.Dense(32, activation="relu"), 
    layers.Dense(10, activation="softmax") 
])

model.compile(optimizer="rmsprop",\
loss="sparse_categorical_crossentropy", \
metrics=["accuracy"])
    
history = model.fit(x_train, \
y_train, \
epochs = 300)    

results = model.predict(x_test)
print(results)

r = sr.Recognizer()
m = sr.Microphone()

os.system("del wake.mp3")
os.system("del confusion.mp3")
os.system("del result.mp3")

myobj = gTTS("Yes?", lang='en', slow=False)
myobj.save("wake.mp3")

myobj = gTTS("Sorry, I couldn't understand you.", lang='en', slow=False)
myobj.save("confusion.mp3")

try:
    file1 = open(r"C:\Users\nokil\OneDrive\Documents\College\Brockport Spring 2023\Classes\Artifical Intell\Project\voice_assistance\data.txt", "r")
    data_exist = True
except:
    data_exist = False
    print("Couldn't find data.txt")

while True:
    with m as source:
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source, phrase_time_limit=2)
    try:
        text=r.recognize_google(audio)
        wakeIndex = text.find('doctor you')
        print(text)
        if wakeIndex >= 0:
            playsound("wake.mp3")
            print("Speak....")
            with m as source: audio = r.listen(source, phrase_time_limit=5)
            try:
                text=r.recognize_google(audio)
                text = text.replace("+", "plus")
                text = text.replace("-", "minus")
                text = text.replace("*", "times")
                text = text.replace("/", "divided by")
                print("You said-> {}".format(text))
                
                if text == "stop":
                    break
                elif text == "cancel":
                    continue
                else:
                    textList = text.split(" ")
                    textListNums = textList.copy()
                    
                    for i in range(0, len(textList)):
                        if textList[i].isdigit():
                            textListNums[i] = word_index["number"]
                            continue
                        elif textList[i] in word_index:
                            textListNums[i] = word_index[textList[i]]
                        else:
                            textListNums[i] = word_index["unknown"]
                    
                    vectorizedData = np.zeros((1, dimension))
                    
                    for i in textListNums:
                        vectorizedData[0, i] = 1.
                    
                    results = model.predict(vectorizedData).tolist()
                    print(results)

                    if data_exist == True:
                        for line in file1:
                            pass
                        last_line = line
                        last_line = last_line.strip("\n")
                        tempHum = last_line.split(",")
                    
                    maxIndex = results[0].index(max(results[0]))
                    
                    if max(results[0]) < 0.75:
                        playsound("confusion.mp3")
                        continue
                    
                    match maxIndex:
                        case 0:
                            if data_exist == True:
                                speech = 'Sure, the temperature is ' + tempHum[0] + ' degrees'
                            else:
                                speech = "Sorry, I don't have access to temperature data"
                            myobj = gTTS(speech, lang='en', slow=False)
                            myobj.save("result.mp3")
                            playsound("result.mp3")
                        
                        case 1:
                            if data_exist == True:
                                speech = 'Sure, the humidity is ' + tempHum[1] + ' percent'
                            else:
                                speech = "Sorry, I don't have access to humidity data"
                            myobj = gTTS(speech, lang='en', slow=False)
                            myobj.save("result.mp3")
                            playsound("result.mp3")
                        
                        case 2:
                            nums = get_nums(textList)
                            if len(nums) != 2:
                                speech = 'I need two numbers to add together'
                                myobj = gTTS(speech, lang='en', slow=False)
                                myobj.save("result.mp3")
                                playsound("result.mp3")
                                continue
                            else:
                                num1 = float(nums[0])
                                num2 = float(nums[1])
                                total = str(num1 + num2)
                                speech = 'Sure, ' + nums[0] + ' plus ' + nums[1] + ' is ' + total
                                myobj = gTTS(speech, lang='en', slow=False)
                                myobj.save("result.mp3")
                                playsound("result.mp3")
                            
                        case 3:
                            nums = get_nums(textList)
                            if len(nums) != 2:
                                speech = 'I need two numbers to subtract'
                                myobj = gTTS(speech, lang='en', slow=False)
                                myobj.save("result.mp3")
                                playsound("result.mp3")
                                continue
                            else:
                                num1 = float(nums[0])
                                num2 = float(nums[1])
                                total = str(num1 - num2)
                                speech = 'Sure, ' + nums[0] + ' minus ' + nums[1] + ' is ' + total
                                myobj = gTTS(speech, lang='en', slow=False)
                                myobj.save("result.mp3")
                                playsound("result.mp3")
                                
                        case 4:
                            nums = get_nums(textList)
                            if len(nums) != 2:
                                speech = 'I need two numbers to multiply'
                                myobj = gTTS(speech, lang='en', slow=False)
                                myobj.save("result.mp3")
                                playsound("result.mp3")
                                continue
                            else:
                                num1 = float(nums[0])
                                num2 = float(nums[1])
                                total = str(num1 * num2)
                                speech = 'Sure, ' + nums[0] + ' times ' + nums[1] + ' is ' + total
                                myobj = gTTS(speech, lang='en', slow=False)
                                myobj.save("result.mp3")
                                playsound("result.mp3")
                                
                        case 5:
                            nums = get_nums(textList)
                            if len(nums) != 2:
                                speech = 'I need two numbers to divide'
                                myobj = gTTS(speech, lang='en', slow=False)
                                myobj.save("result.mp3")
                                playsound("result.mp3")
                                continue
                            else:
                                if nums[1] == "0":
                                    speech = "Do I sound like a fool to you?"
                                else:
                                    num1 = float(nums[0])
                                    num2 = float(nums[1])
                                    total = str(num1 / num2)
                                    speech = 'Sure, ' + nums[0] + ' divided by ' + nums[1] + ' is ' + total
                                myobj = gTTS(speech, lang='en', slow=False)
                                myobj.save("result.mp3")
                                playsound("result.mp3")
                            
                        case 6:
                            speech = "I’m a simple voice assistant capable of telling you the time, playing music from youtube, temperature and humidity of my environment, and the answers to basic mathematical questions. I was created by 3 undergraduate students with nothing better to do with their time."
                            myobj = gTTS(speech, lang='en', slow=False)
                            myobj.save("result.mp3")
                            playsound("result.mp3")
                        
                        case 7:
                            now = datetime.now()
                            current_time = now.strftime("%I:%M %p")
                            speech = 'Sure, the time is ' + current_time
                            
                            myobj = gTTS(speech, lang='en', slow=False)
                            myobj.save("result.mp3")
                            playsound("result.mp3")
                        
                        case 8:
                            if "by" in text:
                                text = text.split("by", 1)[1].lstrip()

                                speech = 'Sure, one second while I play' + text.replace("me","you")
                                myobj = gTTS(speech, lang='en', slow=False)
                                myobj.save("result.mp3")
                                playsound("result.mp3")

                                pywhatkit.playonyt(text)

                            elif "play" in text:
                                text = text.split("play", 1)[1].lstrip()
                                speech = 'Sure, one second while I play' + text.replace("me","you")
                                myobj = gTTS(speech, lang='en', slow=False)
                                myobj.save("result.mp3")
                                playsound("result.mp3")
                                pywhatkit.playonyt(text)
                        
                        case 9:
                            webbrowser.open("https://sites.google.com/view/ningyu")
                                
                        case _:
                            print("default")
                            
                    os.system("del result.mp3")

            except Exception as e:
                print("Sorry couldn't hear you")
                playsound("confusion.mp3")
                print(e)
        else:
            wakeIndex = text.find('stop')
            if wakeIndex >= 0:
                break
    except:
        continue
        
os.system("del result.mp3")
os.system("del wake.mp3")
os.system("del confusion.mp3")

file1.close()

