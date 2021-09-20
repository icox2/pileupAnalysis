#Goal is to take in a file(s) for trace analysis and determine number of pileups within a single event (ML? Derivative?)
#Eventually try to incorporate into PAASS or have experiment online plot
#Currently this is written in python for the hopes of using ML.. if ML is not the best option this might need to be rewritten in cpp
from numpy.lib.npyio import save
import tensorflow as tf
import ROOT
import sys
import os
import csv
import numpy as np
import pandas as pd
from array import array
import matplotlib.pyplot as plt
import random

#Functions to execute on the traces
def bSub(trace):
    if len(trace)==0:
        return trace
    baseline = float()
    for i in range(0,20):
        baseline += trace[i]
    baseline /=20
    max = int(0)
    tm = 0
    for i in range(len(trace)):
        trace[i]-=baseline
        if trace[i]>trace[max]:
            max = i
            #print(max, trace[max], i)
    tm = trace[max]
    trace = [x * (1/tm) for x in trace]
    return trace

def der(trace):
    if len(trace)==0:
        return trace
    der = [0.]
    for it in range(0,len(trace)-1):
        der.append(trace[it+1]-trace[it])
    return der

outHistFile = ROOT.TFile.Open("corr.root" ,"RECREATE")
#Histogram and Tree for later
corr = ROOT.TH2F("correct","Correctness;Prediction;Actual",5,0,5,5,0,5)
miss = ROOT.TH2F("miss","Missed EventNum",10000,0,10000,5,0,5)
ctree = ROOT.TTree("ctree","tree used to analyze NN")

#Want to take in the training data, training labels and the test file.  
if len(sys.argv) != 4:
    print("Usage: %s <train label file> <train data file> <test input file> ..."%(sys.argv[0]))
    sys.exit(1)

train_data = []
train_label = []

#This loop is used to extract and format the inpout data and labels
for iter in range(2):
    inFileName = sys.argv[iter+1]
    print("Reading from",inFileName)
    splitName = os.path.splitext(inFileName)
    fileExt = splitName[1]
    trace = []
    """if fileExt=='.root':
        inFile = ROOT.TFile.Open(inFileName,"READ")
        tree = inFile.Get("newTree")
        rootBool = True"""
    if iter==0:
        with open(inFileName) as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=",")
            count = 0
            #Loop through all of the events to get the various branches which we want to analyze
            for row in csv_reader:
                val = int(row[0])
                train_label.append(val)
    else:
        #inFile = open(inFileName)
        with open(inFileName,'r',newline='') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=",")
            count = 0
            #Loop through all of the events to get the various branches which we want to analyze
            for row in csv_reader:
                count += 1
                trace = list(map(int,row))
                """i = 0
                for iter in row:
                    trace.append(int(float(row[i])))
                    i += 1"""
                noBase = np.array(bSub(trace))
                length = len(noBase)
                train_data.append(np.array(der(noBase)))
        
    
    """ for entryNum in range(0,tree.GetEntries()):
        tree.GetEntry(entryNum)
        trace = list(getattr(tree,"dynodeTraceh"))
        baseline = getattr(tree,"baseline")
        noBase = np.array(bSub(trace,baseline))
        if iter==0:
            length = len(noBase)
        elif len(noBase)>=length:
            length = len(noBase)
        else:
            for i in range(len(noBase),length):
                noBase = np.append(noBase,0)
        
        if len(noBase)==0:
            train_label.append(0)
            train_data.append(np.zeros(500))
            continue 
        elif iter==0:
            train_label.append(1)
        elif (iter==1 and entryNum<5448):
            train_label.append(3)
        else:
            train_label.append(2)
        if iter==0:
            train_label.append(2)
        else:
            train_label.append(1)
        train_data.append(np.array(noBase))"""
        

    #inFile.Close()
#Simple sanity checks
print(len(train_data), type(train_data), len(train_label), type(train_label), 'noBaseLow', type(noBase))
if len(train_data)==len(train_label):
    print("Same Length")
else:
    print("!! Different Lengths !!")

class_names = ['No Pulse', 'Single Pulse', 'Double Pulse']
#Defining the neural network layers
model = tf.keras.Sequential([
    tf.keras.layers.Dense(5, activation='relu'),
    #tf.keras.layers.Dense(5, activation='relu'),
    tf.keras.layers.Dense(3)
])

model.compile(optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

train_data = np.array(train_data)
train_label = np.array(train_label)

model.fit(train_data, train_label, epochs=10)

#Getting and formatting test data
#This might need to be generalized in the future
test_data = []
test_label = []
test_phase = []
inFileName = sys.argv[3]
print("Reading from",inFileName)
inFile = ROOT.TFile.Open(inFileName,"READ")
"""tree = inFile.Get("newTree")
#Loop through all of the events to get the various branches which we want to analyze
for entryNum in range(0,tree.GetEntries()):
    if entryNum==20000:
        break
    tree.GetEntry(entryNum)
    trace = list(getattr(tree,"dynodeTraceh"))
    #baseline = getattr(tree,"baseline")
    baseline = getattr(tree,"bDyn")
    #baseline = 830
    noBase = np.array(bSub(trace,baseline))
    if len(noBase)==0:
        test_label.append(np.array([0]))
        test_data.append(np.zeros(500))
        continue
    elif len(noBase)> length:
        #print("Test Data Traces are long.  Removing end")
        del noBase[length:]
        test_label.append(np.array([2]))
    elif len(noBase)<length:
        for i in range(len(noBase),length):
            noBase = np.append(noBase,0)
        test_label.append(np.array([2]))
    else:
        test_label.append(np.array([2]))
    test_data.append(np.array(noBase))"""
        
tree = inFile.Get("timing") #Tree Name
for entryNum in range(0,tree.GetEntries()):
    if not(entryNum%1000):
        print(entryNum, end='\r')
    if entryNum == 100000:
        break
    tree.GetEntry(entryNum)
    trace = list(getattr(tree,"traceone"))
    randPile = bool(random.getrandbits(1))
    randPhase = random.randint(7,60)
    if max(trace)<10000 and max(trace)>4000 and randPile:
        noBase = np.array(bSub(trace))
        i=1
        outRange = True
        noBase1 = [0] * randPhase
        while outRange:
            tree.GetEntry(entryNum+i)
            trace1 = list(getattr(tree,"traceone"))
            if max(trace1)<10000 and max(trace1)>4000:
                noBase1.extend(list(bSub(trace1)))
                del noBase1[len(noBase):]
                #print(noBase1)
                sum =  [a+b for a,b in zip(noBase,noBase1)]#this needs to be fixed to include random phase
                #print(sum)
                outRange = False
            elif i>5000 or i+entryNum>tree.GetEntries():
                print('\nValue too far', i+entryNum)
                break
            i += 1
        if len(sum)<length:
            x0 = len(sum)-1
            y0 = sum[x0-1]
            slope = (0.-y0)/(length-x0)
            inter = y0 - slope*x0
            for i in range(len(sum),length):
                sum = np.append(sum,slope*i+inter)
        test_data.append(der(sum))
        test_label.append(np.array([2]))
        test_phase.extend(np.array([randPhase]))
    elif max(trace)<10000 and max(trace)>4000 and not randPile:
        noBase = np.array(bSub(trace))
        if len(noBase)<length:
            x0 = len(noBase)-1
            y0 = noBase[x0-1]
            slope = (0.-y0)/(length-x0)
            inter = y0 - slope*x0
            for i in range(len(noBase),length):
                noBase = np.append(noBase,slope*i+inter)
        test_data.append(der(noBase))
        test_label.append(np.array([1]))
        test_phase.append(np.array([0]))
    else:
        continue
        noBase = np.array(bSub(trace,735))
        if len(noBase)<length:
            for i in range(len(noBase),length):
                noBase = np.append(noBase,0)
        test_data.append(noBase)
        test_label.append(np.array([0]))

inFile.Close()

#Section for using the test data to test the model
if len(test_data)==len(test_label):
    print("Tests Same Length", len(test_data))
else:
    print("!! Tests Different Lengths !!")

print(len(test_data), type(test_data), len(test_label), type(test_label))
test_data = np.array(test_data)
test_label = np.array(test_label)

test_loss, test_acc = model.evaluate(test_data,  test_label, verbose=2)
print('\nTest accuracy:', test_acc)

probability_model = tf.keras.Sequential([model,  tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_data)
predictions[0]
np.argmax(predictions[0])

#Section for writing the output file used for analysis
length = len(test_data[0])
print(length)
save_data = array('f',500*[0.])
ctree.Branch("trace",save_data,"trace[500]/F")
#ctree.Branch("trace", save_data, 'TString::Format(“trace[%i]/F”, length)')
save_label = array('i',[0])
save_evnt = array('i',[0])
save_phase = array('i',[0])
save_predict = array('i', [0])
ctree.Branch("label", save_label, 'label/I')
ctree.Branch("eNum", save_evnt, 'eNum/I')
ctree.Branch("predict", save_predict, 'predict/I')
ctree.Branch("phaes", save_phase, 'phase/I')
for j in range(0,len(test_label)):
    if not(j%1000):
        print(j, end='\r')
    if np.argmax(predictions[j])!=test_label[j]:
        miss.Fill(j,np.argmax(predictions[j]))
    corr.Fill(np.argmax(predictions[j]),test_label[j])
    for k in range(0,len(test_data[j])):
        save_data[k] = test_data[j][k]
    #if j==1:
    #    print(save_data, test_data)
    save_label[0] = test_label[j][0]
    save_phase[0] = 2*int(test_phase[j])
    save_predict[0] = np.argmax(predictions[j])
    save_evnt[0] = j
    ctree.Fill()
miss.SetDirectory(0)
corr.SetDirectory(0)
ctree.SetDirectory(0)

#Python output generally not used
def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.plot(img)
  
  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  print("True Labels", true_label[0], "Predicted Labels", predicted_label)
  #plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
  #                              100*np.max(predictions_array)),
  #                              class_names[true_label[0]],
  #                              color=color)

def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')
  
i = 10
#plt.figure(figsize=(6,3))
#plt.figure()
#plot_image(i, predictions[i], test_labelTens, test_dataTens)
#plt.subplot(1,2,2)
#plot_value_array(i, predictions[i],  test_label)
#plt.savefig("ml.png")
#plt.show()

outHistFile.cd()
corr.Write()
miss.Write()
ctree.Write()
outHistFile.Close()