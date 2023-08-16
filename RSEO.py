# Implementation of Reward-Storage-first Energy-then Optimization (RSEO) algorithm
# No storage constraint, so perform min set cover on all sensors/hoverpoints

from mapping import plotPath, getSensorNames
import matplotlib.pyplot as plt
from ML import getMSE
import random
import numpy as np
from algorithmHelpers import processFiles, getEnergy, remBestHP, createMSEPlot, updateMSEPlot, printResults, \
    writeResults


def RSEO(savePlots=False, pathPlotName="", msePlotName="", outputTextName="", droneHeight=15,
         energyWeight=0, communicationRadius=70, energyBudget=60000, joulesPerMeter=10,
         joulesPerSecond=35, dataSize=250, transferRate=9, minSet=True, generate=False, numSensors=37):
    fig1, ax1, HP_gdf, selected, unselected, sensorNames, df = processFiles(droneHeight, communicationRadius, minSet,
                                                                        generate, numSensors)
    x, y, line, ax2, fig2 = createMSEPlot()
    selected, distance = getEnergy(selected, sensorNames, joulesPerMeter, joulesPerSecond, dataSize, transferRate)
    loopCount = 0
    while selected['energy'][0] > energyBudget:
        loopCount += 1
        unselected, selected = remBestHP(unselected, selected, sensorNames=sensorNames, df=df,
                             energyWeight=energyWeight, joulesPerMeter=joulesPerMeter, joulesPerSecond=joulesPerSecond,
                             dataSize=dataSize, transferRate=transferRate)
        selected, distance = getEnergy(selected, sensorNames, joulesPerMeter, joulesPerSecond, dataSize, transferRate)
        features = getSensorNames(selected['geometry'], sensorNames)
        mse = getMSE(features, df)
        updateMSEPlot(loopCount, mse, ax2, fig2, x, y, line)
        print(f"{selected['energy'][0]} joules")
        print(f"MSE: {mse}")
    printResults(finalSHP=selected, finalDistance=distance, finalIteration=loopCount, finalMSE=mse, sensorNames=sensorNames)
    plotPath(ax1, selected)
    for key, value in sensorNames.items():
            ax1.text(key.x, key.y, str(value), fontsize=10, ha='center', va='bottom')
    if savePlots:
        writeResults(selected, loopCount, distance, mse, sensorNames, outputTextName)
        fig1.savefig(pathPlotName, bbox_inches='tight')
        fig2.savefig(msePlotName, bbox_inches='tight')
        return mse
    else:
        plt.show()

rSeed = 1
random.seed(rSeed)
np.random.seed(rSeed)
RSEO()
