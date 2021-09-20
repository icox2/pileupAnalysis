#dataImport.py
#Used to import and process data into the form best used for ML

import ROOT
import sys
import os
import csv

def rootManage():
    something = 1

def dataImport(inFileName):
    print("Reading from",inFileName)
    splitName = os.path.splitext(inFileName)
    fileExt = splitName[1]
    if fileExt=='.root':
        inFile = ROOT.TFile.Open(inFileName,"READ")
        tree = inFile.Get("newTree")
        rootBool = True
    else:
        inFile = open(inFileName)
        with open(inFileName,newline='') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=",")
            count = 0
            for row in csv_reader:
                trace = row
                count += 1
