import pandas as pd

from iCarl import *
from GDump import *
from LWF import *
from dataSets import *
import os


### Cpu / Gpu
### ### ### ### ### ### ### ### ### ### ### ### ### ###
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



### Parameters
### ### ### ### ### ### ### ### ### ### ### ### ### ###
dataSetNams = ["audioCub","ESC10","ESC50"]

RunConfigFile = "GDumb_Config.csv"
strategy = "GDumb"

#RunConfigFile = "ICARL_Config_ESC50.csv"
#strategy = "Gdumb"


### Results Folder
### ### ### ### ### ### ### ### ### ### ### ### ### ###
fileExist = True
i = -1
while(fileExist):
    i = i + 1
    baseSaveResultsDir = "Results_{:d}/".format(i)
    fileExist = os.path.exists(baseSaveResultsDir)


os.makedirs(baseSaveResultsDir)


configAllDF = pd.read_csv(RunConfigFile)
configAllDF["trainAccMean"] = -1
configAllDF["trainAccLast"] = -1
configAllDF["testAcc"] = -1
configAllDF["procTime"] = -1

CUDA_LAUNCH_BLOCKING=1
### training
### ### ### ### ### ### ### ### ### ### ### ### ### ###
for i in range(len(configAllDF)):
    config = configAllDF.iloc[i].to_dict()
    print("Config ", i, " is ",config)

    if(strategy == "ICARL"):
        trainAccMean , trainAccLast, testAcc , procTime= runiCarl(config,baseSaveResultsDir,i)
    elif(strategy == "LWF"):
        trainAccMean , trainAccLast, testAcc , procTime= runLWF(config,baseSaveResultsDir,i)

    else:
        trainAccMean , trainAccLast, testAcc , procTime= runGDUMB(config,baseSaveResultsDir,i)

    configAllDF.at[i, 'trainAccMean'] = trainAccMean
    configAllDF.at[i, 'trainAccLast'] = trainAccLast
    configAllDF.at[i, 'testAcc'] = testAcc
    configAllDF.at[i, 'procTime'] = procTime

    configAllDF.to_csv(baseSaveResultsDir+strategy+"L_Config_with_Results.csv",index=False)

