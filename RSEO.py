# Implementation of Reward-Storage-first Energy-then Optimization (RSEO) algorithm
# No storage constraint, so perform min set cover on all sensors/hoverpoints



from mapping import processSHP, findMinTravelDistance, plotPath, getSensorNames
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from ML import getMSE, processData
import random
from joblib import Parallel, delayed
import time
import numpy as np
import SetCoverPy
from algorithmHelpers import processFiles, getEnergy, getPointsInBudget, addRandomHP, remRandomHP, addBestHP, remBestHP, \
    createMSEPlot, updateMSEPlot, printResults, writeResults, printTime

np.random.seed(42)
random.seed(42)

def RSEO(savePlots=False, pathPlotName="", msePlotName="", outputTextName="", droneHeight=15,
                  rewardMode="NORMALIZED", thresholdFraction=0.95, energyWeight=0,
                  communicationRadius=70, energyBudget=60000, joulesPerMeter=10, joulesPerSecond=35, dataSize=250,
                  transferRate=9, minSet=True):
    fig1, ax1, allHoverPoints, selected, unselected, sensorNames, df = processFiles(height=droneHeight, communicationRadius=communicationRadius, minSet=minSet)
    x, y, line, ax2, fig2 = createMSEPlot()
    selected, distance = getEnergy(selected, sensorNames, joulesPerMeter, joulesPerSecond, dataSize, transferRate)
    loopCount = 0
    while selected['energy'][0] > energyBudget:
        loopCount += 1
        unselected, selected = remBestHP(unselected, selected, rewardMode=rewardMode, sensorNames=sensorNames, df=df,
                             energyWeight=energyWeight, joulesPerMeter=joulesPerMeter, joulesPerSecond=joulesPerSecond,
                             dataSize=dataSize, transferRate=transferRate, thresholdFraction=thresholdFraction)
        selected, distance = getEnergy(selected, sensorNames, joulesPerMeter, joulesPerSecond, dataSize, transferRate)
        features = getSensorNames(selected['geometry'], sensorNames)
        mse = getMSE(features, df)
        updateMSEPlot(loopCount, mse, ax2, fig2, x, y, line)
        print(f"{selected['energy'][0]} joules")
        print(f"MSE: {mse}")
    printResults(finalSHP=selected, finalDistance=distance, finalIteration=loopCount, finalMSE=mse, sensorNames=sensorNames)
    plotPath(ax1, selected)
    if savePlots:
        writeResults(selected, loopCount, distance, mse, sensorNames, outputTextName)
        fig1.savefig(pathPlotName, bbox_inches='tight')
        fig2.savefig(msePlotName, bbox_inches='tight')
        return mse
    else:
        plt.show()

# for i in range(10):
#     pathPlotName = f"Experiments/RSEO/path{i + 1}"
#     msePlotName = f"Experiments/RSEO/mse{i + 1}"
#     outputTextName = f"Experiments/RSEO/output{i + 1}.txt"
#     RSEO(savePlots=True, pathPlotName=pathPlotName, msePlotName=msePlotName, outputTextName=outputTextName)