#autoencoder.py
#used to reduce the amount of noice in the traces.

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,losses
from tensorflow.keras.models import Model
import ROOT
import sys
import os
import csv
import numpy as np
import random
from array import array
import matplotlib.pyplot as plt
from pileupFunctions import bSub,der,normalize

if len(sys.argv)>3:
  print("Incorrect Number of Files")

#get the input traces
train_data = []
reg_trace = []
first_trace = []
phase = []
inFileName = sys.argv[1]
print("Reading from",inFileName)
inFile = ROOT.TFile.Open(inFileName,"READ")
        
tree = inFile.Get("timing") #Tree Name
for entryNum in range(0,tree.GetEntries()):
  if not(entryNum%1000):
    print("train data",entryNum, end='\r')
  if entryNum==50000:
    break
  tree.GetEntry(entryNum)
  trace = list(getattr(tree,"traceone"))
  #randPile = bool(random.randint(0,4))
  randPile = False
  randPhase = random.randint(0,100)
  if max(trace)>3000 and not randPile:
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
        sum =  [a+b for a,b in zip(noBase,noBase1)]#this needs to be fixed to include 
        noNorm = sum
        sum = normalize(sum)
        outRange = False
      elif i>5000 or i+entryNum>tree.GetEntries():
        print('\nValue too far', i+entryNum)
        break
      i += 1
    train_data.append(der(sum))
    reg_trace.append(sum)
    first_trace.append(noNorm)
    phase.append(randPhase)
  elif max(trace)>3000:
    noBase = np.array(normalize(trace))
    if len(noBase)<10:
      continue
    if len(noBase)<500:
      x0 = len(noBase)-1
      y0 = noBase[x0-1]
      slope = (0.-y0)/(length-x0)
      inter = y0 - slope*x0
      for i in range(len(noBase),length):
        noBase = np.append(noBase,slope*i+inter+random.randint(-100,100))
    first_trace.append(bSub(trace))
    train_data.append(der(noBase))
    reg_trace.append(noBase)
    phase.append(0)
  else:
    continue
inFile.Close()


#defining the Autoencoder Model class
class Autoencoder(Model):
  def __init__(self):
    super(Autoencoder, self).__init__()
    self.encoder = tf.keras.Sequential([
      layers.Dense(45, activation='relu'),
      layers.Dense(11, activation='relu'),
    ])
    self.decoder = tf.keras.Sequential([
      layers.Dense(45, activation='relu'),
      layers.Dense(500, activation='relu')
    ])
  
  def call(self,x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

checkpoint_path = "models/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
#Creating a call back that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,save_weights_only=True,sverbose=1)

autoencoder = Autoencoder()
autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError(), metrics=['accuracy'])
train_data = np.array(train_data)
print(len(train_data), type(train_data))
autoencoder.fit(train_data,train_data,epochs=40,shuffle=True,callbacks=[cp_callback])

os.listdir(checkpoint_dir)
autoencoder.save("models/autoencode")

decodedTrace = array('f',500*[0.])
originalTrace = array('f',500*[0.])
noDer = array('f',500*[0.])
firTrace = array('f',500*[0.])
ratioTrace = array('f',500*[0.])
encodedData = autoencoder.encoder(train_data).numpy()
decodedData = autoencoder.decoder(encodedData).numpy()

print(len(decodedData), type(decodedData))
save_phase = array('i',[0])
thresh_label = array('i',[0])
avgDev = array('f',[0])

#output
outHistFile = ROOT.TFile.Open("decoded.root" ,"RECREATE")
dtree = ROOT.TTree("dtree","decoded data")
dtree.Branch("tPile",thresh_label,"tPile/I")
dtree.Branch("phase",save_phase,"phase/I")
dtree.Branch("avgDev",avgDev,"avgDev/F")
dtree.Branch("oTrace",originalTrace,"oTrace[500]/F")
dtree.Branch("dTrace",decodedTrace,"dTrace[500]/F")
dtree.Branch("rTrace",noDer,"rTrace[500]/F")
dtree.Branch("fTrace",firTrace,"fTrace[500]/F")
dtree.Branch("raTrace",ratioTrace,"raTrace[500]/F")

for i in range(0,len(train_data)):
  save_phase[0] = phase[i]
  numPile = 0
  avg = 0.
  cnt = 0
  for j in range(0,len(train_data[i])):
    originalTrace[j] = train_data[i][j]
    decodedTrace[j] = decodedData[i][j]
    if originalTrace[j]>0.06:
      ratioTrace[j] = originalTrace[j]-decodedTrace[j];
      avg += ratioTrace[j]
      cnt += 1
    else:
      ratioTrace[j]=0;
    noDer[j] = reg_trace[i][j]
    firTrace[j] = first_trace[i][j]
  for k in range(1,len(originalTrace)-1):
    if originalTrace[k]>0.1 and originalTrace[k]>originalTrace[k-1] and originalTrace[k]>originalTrace[k+1]:
      numPile += 1
  thresh_label[0] = numPile
  avgDev[0] = avg/cnt
  dtree.Fill()

dtree.SetDirectory(0)
outHistFile.cd()
dtree.Write()
outHistFile.Close()

"""def plot_loss(autoencoder):
  plt.plot(autoencoder.autoencoder['loss'], label='loss')
  plt.plot(autoencoder.autoencoder['val_loss'], label='val_loss')
  plt.ylim([0,10])
  plt.xlabel('Epoch')
  plt.ylabel('Error')
  plt.legent()
  plt.grid(True)

plot_loss(autoencoder)"""