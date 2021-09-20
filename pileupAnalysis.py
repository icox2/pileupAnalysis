#Goal is to take in a file(s) for trace analysis and determine number of pileups within a single event (ML? Derivative?)
#Eventually try to incorporate into PAASS or have experiment online plot
#Currently this is written in python for the hopes of using ML.. if ML is not the best option this might need to be rewritten in cpp
import tensorflow as tf
import ROOT
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def bSub(trace, baseline):
    if len(trace)==0:
        return trace
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

#Histogram for later
corr = ROOT.TH2F("correct","Correctness;Prediction;Actual",5,0,5,5,0,5)
miss = ROOT.TH1F("miss","Missed EventNum",10000,0,10000)

#Right now we can just pass in a file (or more).. this might be changed later with larger datasets.
if len(sys.argv) != 4:
    print("Usage: %s <train input file> ... <test input file> ..."%(sys.argv[0]))
    sys.exit(1)

train_data = []
train_label = []

for iter in range(2):

    inFileName = sys.argv[iter+1]
    print("Reading from",inFileName)

    inFile = ROOT.TFile.Open(inFileName,"READ")
    tree = inFile.Get("newTree")

    #Loop through all of the events to get the various branches which we want to analyze
    for entryNum in range(0,tree.GetEntries()):
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
#        elif iter==0:
#            train_label.append(1)
#        elif (iter==1 and entryNum<5448):
#            train_label.append(3)
#        else:
#            train_label.append(2)
        if iter==0:
            train_label.append(2)
        else:
            train_label.append(1)
        train_data.append(np.array(noBase))
        

    inFile.Close()

print(len(train_data), type(train_data), len(train_label), type(train_label), 'noBaseLow', type(noBase))
if len(train_data)==len(train_label):
    print("Same Length")
else:
    print("!! Different Lengths !!")

class_names = ['No Pulse', 'Single Pulse', 'Double Pulse', 'Triple Pulse', 'Four Pulses']

model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu'),
    #tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(5)
])

model.compile(optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

train_data = np.array(train_data)
train_label = np.array(train_label)

model.fit(train_data, train_label, epochs=10)

#getting test data
test_data = []
test_label = []
inFileName = sys.argv[3]
print("Reading from",inFileName)
inFile = ROOT.TFile.Open(inFileName,"READ")
tree = inFile.Get("newTree")
#Loop through all of the events to get the various branches which we want to analyze
for entryNum in range(0,tree.GetEntries()):
    tree.GetEntry(entryNum)
    trace = list(getattr(tree,"dynodeTraceh"))
    baseline = getattr(tree,"bDyn")
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
    test_data.append(np.array(noBase))
        
inFile.Close()

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


for j in range(entryNum):
    #print(j)
    if np.argmax(predictions[j])!=test_label[j]:
        miss.Fill(j)
    corr.Fill(np.argmax(predictions[j]),test_label[j])
miss.SetDirectory(0)
corr.SetDirectory(0)

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

outHistFile = ROOT.TFile.Open("corr.root" ,"RECREATE")
outHistFile.cd()
corr.Write()
miss.Write()
outHistFile.Close()