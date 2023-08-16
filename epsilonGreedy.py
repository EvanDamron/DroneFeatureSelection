# epsilon squared greedy algorithm to find the optimal set of hover points to predict the other features without
# violating battery-of-drone budget
# Code Written by Evan Damron, University of Kentucky Computer Science

from mapping import findMinTravelDistance, plotPath, getSensorNames
import pandas as pd
import matplotlib.pyplot as plt
from ML import getMSE
import random
import time
import numpy as np
from algorithmHelpers2 import processFiles, getEnergy, getPointsInBudget, addRandomHP, remRandomHP, addBestHP, remBestHP, \
    createMSEPlot, updateMSEPlot, printResults, writeResults, printTime



def epsilonGreedy(numLoops, startTime, savePlots=False, pathPlotName="", msePlotName="", outputTextName="",
                  droneHeight=15, energyWeight=0, communicationRadius=70, energyBudget=60000, joulesPerMeter=10,
                  joulesPerSecond=35, dataSize=250, transferRate=9, minSet=False, generate=False, numSensors=37):

    fig1, ax1, HP_gdf, UHP_gdf, SHP_gdf, sensorNames, df = processFiles(droneHeight, communicationRadius, minSet,
                                                                        generate, numSensors)
    print(f"Total number of Hoverpoints: {len(HP_gdf)}")
    x, y, line, ax2, fig2 = createMSEPlot()
    print(f"EPSILON-GREEDY ALGORITHM WITH {numLoops} LOOPS AND ENERGY WEIGHT {energyWeight}")
    arProb = 0  # probability of adding (0) and removing (1)
    rbProb = 1  # probability of random (1) and best (0)
    loopCount = 0
    pointsInBudget = getPointsInBudget(UHP_gdf, SHP_gdf, sensorNames, energyBudget, joulesPerMeter, joulesPerSecond,
                                       dataSize, transferRate)

    minMSE = 1000

    # Make sure every hover-point in hp is accounted for in uhp and shp
    if len(UHP_gdf) + len(SHP_gdf) != len(HP_gdf):
        print('ERROR: SELECTED + UNSELECTED != HP')
        exit(1)
    while loopCount < numLoops:
        loopCount += 1
        # energyWeight = 1 - (loopCount / numLoops)
        print("Loop iteration ", loopCount, ' of ', numLoops)
        raProb = rbProb * (1 - arProb)  # random add
        rrProb = rbProb * arProb  # random remove
        baProb = (1 - rbProb) * (1 - arProb)  # best add
        brProb = (1 - rbProb) * arProb  # best remove
        randomNumber = (random.random())
        if randomNumber < raProb:
            UHP_gdf, SHP_gdf = addRandomHP(UHP_gdf, pointsInBudget, SHP_gdf)
        elif randomNumber < raProb + rrProb:
            UHP_gdf, SHP_gdf = remRandomHP(UHP_gdf, SHP_gdf)
        elif randomNumber < raProb + rrProb + baProb:
            UHP_gdf, SHP_gdf = addBestHP(unselected=UHP_gdf, unselectedIB=pointsInBudget, selected=SHP_gdf,
                                         sensorNames=sensorNames, df=df, energyWeight=energyWeight)
        else:
            UHP_gdf, SHP_gdf = remBestHP(UHP_gdf, SHP_gdf, sensorNames=sensorNames, df=df, energyWeight=energyWeight,
                                         joulesPerMeter=joulesPerMeter, joulesPerSecond=joulesPerSecond,
                                         dataSize=dataSize, transferRate=transferRate)

        UHP_gdf = UHP_gdf.reset_index(drop=True)
        SHP_gdf = SHP_gdf.reset_index(drop=True)
        rbProb = rbProb - 1 / numLoops
        features = getSensorNames(SHP_gdf['geometry'], sensorNames)
        mse = getMSE(features, df)
        if mse < minMSE:
            minMSE = mse
            bestSHP = SHP_gdf.copy()
            iterationOfBest = loopCount
        if SHP_gdf.empty:
            totalEnergy = 0
        else:
            SHP_gdf, _ = getEnergy(dataSize=dataSize, joulesPerMeter=joulesPerMeter, joulesPerSecond=joulesPerSecond,
                                   selected=SHP_gdf, sensorNames=sensorNames, transferRate=transferRate)
            totalEnergy = SHP_gdf['energy'][0]

        print(f"This set of hoverpoints requires {totalEnergy} Joules out of the"
              f" {energyBudget} Joules in the drone's battery")
        print('Total Number of Hover Points Visited: ', len(SHP_gdf))
        print(f"current mse: {mse}, lowest mse yet: {minMSE}")
        updateMSEPlot(newX=loopCount, newY=mse, ax=ax2, fig=fig2, x=x, y=y, line=line)
        pointsInBudget = getPointsInBudget(UHP_gdf, SHP_gdf, sensorNames, energyBudget, joulesPerMeter, joulesPerSecond,
                                           dataSize, transferRate)

        if len(UHP_gdf) == 0:
            arProb = totalEnergy / energyBudget
        else:
            arProb = (totalEnergy / energyBudget) * (
                    1 - (len(pointsInBudget) / len(UHP_gdf)))  # as we approach budget, more likely to remove
        if len(UHP_gdf) + len(SHP_gdf) != len(HP_gdf):
            print('ERROR: SELECTED + UNSELECTED != HP')
            print(len(SHP_gdf), '+', len(UHP_gdf), ' != ', len(HP_gdf))
            exit(1)
    # get the cheapest path of best shp, because TSP is heuristic and gives different paths, some better than others
    minEnergy = 999999999999
    for i in range(10):
        tempBestSHP = bestSHP.copy()
        tempBestSHP, _ = findMinTravelDistance(tempBestSHP, scramble=True)
        tempBestSHP, distance = getEnergy(tempBestSHP, sensorNames, joulesPerMeter, joulesPerSecond, dataSize,
                                          transferRate)
        print(f"energy of best {i}: {tempBestSHP['energy'][0]}")
        if tempBestSHP['energy'][0] < minEnergy:
            minEnergy = tempBestSHP['energy'][0]
            bestSHP = tempBestSHP.copy()
            distanceOfBest = distance
    print(f"EPSILON-GREEDY ALGORITHM WITH {numLoops} LOOPS AND ENERGY WEIGHT {energyWeight}")
    printResults(bestSHP, iterationOfBest, distanceOfBest, minMSE, sensorNames)
    ax2.scatter(iterationOfBest, minMSE, color='red', zorder=10)
    plotPath(ax1, bestSHP)
    bestSHP.plot(ax=ax1, color='red', markersize=10, alpha=1)
    printTime(startTime)
    for key, value in sensorNames.items():
        if len(value) == 1:
            ax1.text(key.x, key.y, str(value[0]), fontsize=10, ha='center', va='bottom')
    if savePlots:
        writeResults(bestSHP, iterationOfBest, distanceOfBest, minMSE, sensorNames, outputTextName)
        fig1.savefig(pathPlotName, bbox_inches='tight')
        fig2.savefig(msePlotName, bbox_inches='tight')
    else:
        plt.show()

rSeed = 1
np.random.seed(rSeed)
random.seed(rSeed)
startTime = time.time()
epsilonGreedy(numLoops=200, generate=True, startTime=startTime, energyWeight=0, energyBudget=50000)
