# Implementation of Reward-Storage-first Energy-then Optimization (RSEO) algorithm
# No storage constraint, so perform min set cover on all sensors/hoverpoints

from mapping import plotPath, getSensorNames, addSensorsUniformRandom, getHoverPoints, minSetCover
import matplotlib.pyplot as plt
from ML import getMSE, calculate_feature_importance, normalizeData, processData
import random
import numpy as np
from algorithmHelpers import processFiles, getEnergy, remBestHP, createMSEPlot, updateMSEPlot, printResults, \
    writeResults
import time
import geopandas as gpd

# fig1, ax1, HP_gdf, sensorNames, df,
def RSEO(fig1, ax1, HP_gdf, sensorNames, df,
         savePlots=False, pathPlotName="", msePlotName="", outputTextName="", droneHeight=15,
         communicationRadius=70, energyBudget=60000, joulesPerMeter=10, addToOriginal=True,
         joulesPerSecond=35, dataSize=100, transferRate=9, minSet=True, generate=False, numSensors=37):
    startTime = time.time()
    # fig1, ax1, HP_gdf, selected, unselected, sensorNames, df = processFiles(droneHeight, communicationRadius, minSet,
    #                                                                         generate, addToOriginal, numSensors)

    # print(newDF)
    for i in range(5):
        sensorNames, HP_gdf = minSetCover(sensorNames, HP_gdf)
    selected = HP_gdf.copy()
    HP_gdf.plot(ax=ax1, color='yellow', markersize=10, alpha=1)
    sensorImportances = calculate_feature_importance(df)
    hoverPointImportances = dict.fromkeys(sensorNames.keys(), 0)
    for hoverPoint in HP_gdf['geometry']:
        for sensor in sensorNames[hoverPoint]:
            hoverPointImportances[hoverPoint] += sensorImportances[sensor]
    x, y, line, ax2, fig2 = createMSEPlot()
    selected, distance = getEnergy(selected, sensorNames, joulesPerMeter, joulesPerSecond, dataSize, transferRate)
    loopCount = 0
    while selected['energy'][0] > energyBudget:
        loopCount += 1
        lowestScore = min(hoverPointImportances[point] for point in selected['geometry'])
        lowestScorePoints = [key for key, value in hoverPointImportances.items() if value == lowestScore and key in selected['geometry']]
        # if len(lowestScorePoints) > 1:
        #
        #     print('points were tied...')
        #     exit(1)
        lowestScorePoint = lowestScorePoints[0]
        selected = selected[selected['geometry'] != lowestScorePoint]
        selected, distance = getEnergy(selected, sensorNames, joulesPerMeter, joulesPerSecond, dataSize, transferRate)
        # features = getSensorNames(selected['geometry'], sensorNames)
        # mse = getMSE(features, df)
        # mse = 1
        # updateMSEPlot(loopCount, mse, ax2, fig2, x, y, line)
        # if len(selected) == 0:
        #     print(f"{selected['energy'][0]} joules")
        # print(f"MSE: {mse}")
    if len(selected) == 0:
        print('no solution found')
        return 0
    features = getSensorNames(selected['geometry'], sensorNames)
    print(f'features {features}, {len(features)}')
    targets = [name for name in df.columns if name not in features]
    print(f'targets {targets}, {len(targets)}')
    mse = getMSE(features, df)
    printResults(finalSHP=selected, finalDistance=distance, finalIteration=loopCount, finalMSE=mse,
                 sensorNames=sensorNames)
    plotPath(ax1, selected)
    for key, value in sensorNames.items():
            ax1.text(key.x, key.y, str(value), fontsize=10, ha='center', va='bottom')
    if savePlots:
        writeResults(selected, loopCount, distance, mse, sensorNames, outputTextName, startTime, finalMSE2=0)
        fig1.savefig(pathPlotName, bbox_inches='tight')
        fig2.savefig(msePlotName, bbox_inches='tight')
        return mse
    else:
        print(df)
        print(hoverPointImportances)
        plt.show()


# rSeed = 1
# np.random.seed(rSeed)
# random.seed(rSeed)
# RSEO(savePlots=False, generate=True, energyBudget=60000)
# droneHeight=15
# communicationRadius=70
# minSet = True
# generate = False
# addToOriginal = False
# numSensors = 37
# fig1, ax1, HP_gdf, selected, unselected, sensorNames, df = processFiles(droneHeight, communicationRadius, minSet,
#                                                                             generate, addToOriginal, numSensors)
# sensorImportances = calculate_feature_importance(df)
# NEEDS TO RUN
def originalBudgetTests(sensorImportances):
    for i in range(3, 5):
        pathPlotName = f"Experiments/Budget Experiments3/original/RSEO/30k/path{i + 1}"
        msePlotName = f"Experiments/Budget Experiments3/original/RSEO/30k/mse{i + 1}"
        outputTextName = f"Experiments/Budget Experiments3/original/RSEO/30k/output{i + 1}.txt"
        RSEO(sensorImportances=sensorImportances, savePlots=True, generate=False, pathPlotName=pathPlotName, msePlotName=msePlotName,
             energyBudget=30000, outputTextName=outputTextName)
        pathPlotName = f"Experiments/Budget Experiments3/original/RSEO/35k/path{i + 1}"
        msePlotName = f"Experiments/Budget Experiments3/original/RSEO/35k/mse{i + 1}"
        outputTextName = f"Experiments/Budget Experiments3/original/RSEO/35k/output{i + 1}.txt"
        RSEO(sensorImportances=sensorImportances, savePlots=True, generate=False, pathPlotName=pathPlotName, msePlotName=msePlotName,
             energyBudget=35000, outputTextName=outputTextName)
        pathPlotName = f"Experiments/Budget Experiments3/original/RSEO/40k/path{i + 1}"
        msePlotName = f"Experiments/Budget Experiments3/original/RSEO/40k/mse{i + 1}"
        outputTextName = f"Experiments/Budget Experiments3/original/RSEO/40k/output{i + 1}.txt"
        RSEO(sensorImportances=sensorImportances, savePlots=True, generate=False, pathPlotName=pathPlotName, msePlotName=msePlotName,
             energyBudget=40000, outputTextName=outputTextName)
        pathPlotName = f"Experiments/Budget Experiments3/original/RSEO/45k/path{i + 1}"
        msePlotName = f"Experiments/Budget Experiments3/original/RSEO/45k/mse{i + 1}"
        outputTextName = f"Experiments/Budget Experiments3/original/RSEO/45k/output{i + 1}.txt"
        RSEO(sensorImportances=sensorImportances, savePlots=True, generate=False, pathPlotName=pathPlotName, msePlotName=msePlotName,
             energyBudget=45000, outputTextName=outputTextName)
        pathPlotName = f"Experiments/Budget Experiments3/original/RSEO/50k/path{i + 1}"
        msePlotName = f"Experiments/Budget Experiments3/original/RSEO/50k/mse{i + 1}"
        outputTextName = f"Experiments/Budget Experiments3/original/RSEO/50k/output{i + 1}.txt"
        RSEO(sensorImportances=sensorImportances, savePlots=True, generate=False, pathPlotName=pathPlotName, msePlotName=msePlotName,
             energyBudget=50000, outputTextName=outputTextName)
        pathPlotName = f"Experiments/Budget Experiments3/original/RSEO/55k/path{i + 1}"
        msePlotName = f"Experiments/Budget Experiments3/original/RSEO/55k/mse{i + 1}"
        outputTextName = f"Experiments/Budget Experiments3/original/RSEO/55k/output{i + 1}.txt"
        RSEO(sensorImportances=sensorImportances, savePlots=True, generate=False, pathPlotName=pathPlotName, msePlotName=msePlotName,
             energyBudget=55000, outputTextName=outputTextName)
        pathPlotName = f"Experiments/Budget Experiments3/original/RSEO/25k/path{i + 1}"
        msePlotName = f"Experiments/Budget Experiments3/original/RSEO/25k/mse{i + 1}"
        outputTextName = f"Experiments/Budget Experiments3/original/RSEO/25k/output{i + 1}.txt"
        RSEO(sensorImportances=sensorImportances, savePlots=True, generate=False, pathPlotName=pathPlotName, msePlotName=msePlotName,
             energyBudget=25000, outputTextName=outputTextName)
        pathPlotName = f"Experiments/Budget Experiments3/original/RSEO/20k/path{i + 1}"
        msePlotName = f"Experiments/Budget Experiments3/original/RSEO/20k/mse{i + 1}"
        outputTextName = f"Experiments/Budget Experiments3/original/RSEO/20k/output{i + 1}.txt"
        RSEO(sensorImportances=sensorImportances, savePlots=True, generate=False, pathPlotName=pathPlotName, msePlotName=msePlotName,
             energyBudget=20000, outputTextName=outputTextName)

# originalBudgetTests(sensorImportances)
def synthBudgetTests():
    rSeed = 2
    random.seed(rSeed)
    np.random.seed(rSeed)
    dataFolder = 'CAF_Sensor_Dataset_2/caf_sensors/Hourly'
    df = processData(dataFolder)
    sensorsGDF, df = addSensorsUniformRandom(df=df, numSensors=40)
    print(f'len columns = {len(df.columns)}')
    df = normalizeData(df)
    df = df[np.random.permutation(df.columns)]
    print(f'len columns = {len(df.columns)}')
    fig1, ax1 = plt.subplots(figsize=(8, 8))
    ax1.set_xlim(-100, 1300 + 100)
    ax1.set_ylim(-100, 800 + 100)
    HP_gdf, sensorNames = getHoverPoints(sensorsGDF, commRadius=70, ax=ax1, height=15)
    SHP_gdf = gpd.GeoDataFrame()
    SHP_gdf['geometry'] = None
    SHP_gdf = SHP_gdf.set_geometry('geometry')
    UHP_gdf = HP_gdf.copy()
    SHP_gdf.crs = 'EPSG:3857'  # pseudo-mercator
    UHP_gdf.crs = 'EPSG:3857'
    for j in range(3):
        # fig, ax = plt.subplots(figsize=(8, 8))
        # ax.set_xlim(-100, 1300 + 100)
        # ax.set_ylim(-100, 800 + 100)
        # hoverPoints, sensorNames = getHoverPoints(sensorsGDF, commRadius=70, ax=ax, height=15)
        # pathPlotName = f"Experiments/Budget Experiments3/generated/RSEO/20k/map{rSeed}path{j + 1}"
        # msePlotName = f"Experiments/Budget Experiments3/generated/RSEO/20k/map{rSeed}mse{j + 1}"
        # outputTextName = f"Experiments/Budget Experiments3/generated/RSEO/20k/map{rSeed}output{j + 1}.txt"
        # RSEO(HP_gdf=hoverPoints, ax1=ax, df=df, fig1=fig, sensorNames=sensorNames,
        #      savePlots=True, generate=True, pathPlotName=pathPlotName, msePlotName=msePlotName,
        #      energyBudget=20000, outputTextName=outputTextName, numSensors=40, addToOriginal=True)
        # fig, ax = plt.subplots(figsize=(8, 8))
        # ax.set_xlim(-100, 1300 + 100)
        # ax.set_ylim(-100, 800 + 100)
        # hoverPoints, sensorNames = getHoverPoints(sensorsGDF, commRadius=70, ax=ax, height=15)
        # pathPlotName = f"Experiments/Budget Experiments3/generated/RSEO/25k/map{rSeed}path{j + 1}"
        # msePlotName = f"Experiments/Budget Experiments3/generated/RSEO/25k/map{rSeed}mse{j + 1}"
        # outputTextName = f"Experiments/Budget Experiments3/generated/RSEO/25k/map{rSeed}output{j + 1}.txt"
        # RSEO(HP_gdf=hoverPoints, ax1=ax, df=df, fig1=fig, sensorNames=sensorNames,
        #      savePlots=True, generate=True, pathPlotName=pathPlotName, msePlotName=msePlotName,
        #      energyBudget=25000, outputTextName=outputTextName, numSensors=40, addToOriginal=True)
        # fig, ax = plt.subplots(figsize=(8, 8))
        # ax.set_xlim(-100, 1300 + 100)
        # ax.set_ylim(-100, 800 + 100)
        # hoverPoints, sensorNames = getHoverPoints(sensorsGDF, commRadius=70, ax=ax, height=15)
        # pathPlotName = f"Experiments/Budget Experiments3/generated/RSEO/30k/map{rSeed}path{j + 1}"
        # msePlotName = f"Experiments/Budget Experiments3/generated/RSEO/30k/map{rSeed}mse{j + 1}"
        # outputTextName = f"Experiments/Budget Experiments3/generated/RSEO/30k/map{rSeed}output{j + 1}.txt"
        # RSEO(HP_gdf=hoverPoints, ax1=ax, df=df, fig1=fig, sensorNames=sensorNames,
        #      savePlots=True, generate=True, pathPlotName=pathPlotName, msePlotName=msePlotName,
        #      energyBudget=30000, outputTextName=outputTextName, numSensors=40, addToOriginal=True)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(-100, 1300 + 100)
        ax.set_ylim(-100, 800 + 100)
        hoverPoints, sensorNames = getHoverPoints(sensorsGDF, commRadius=70, ax=ax, height=15)
        pathPlotName = f"Experiments/Budget Experiments3/generated/RSEO/35k/map{rSeed}path{j + 4}"
        msePlotName = f"Experiments/Budget Experiments3/generated/RSEO/35k/map{rSeed}mse{j + 4}"
        outputTextName = f"Experiments/Budget Experiments3/generated/RSEO/35k/map{rSeed}output{j + 4}.txt"
        RSEO(HP_gdf=hoverPoints, ax1=ax, df=df, fig1=fig, sensorNames=sensorNames,
             savePlots=True, generate=True, pathPlotName=pathPlotName, msePlotName=msePlotName,
             energyBudget=35000, outputTextName=outputTextName, numSensors=40, addToOriginal=True)
        # fig, ax = plt.subplots(figsize=(8, 8))
        # ax.set_xlim(-100, 1300 + 100)
        # ax.set_ylim(-100, 800 + 100)
        # hoverPoints, sensorNames = getHoverPoints(sensorsGDF, commRadius=70, ax=ax, height=15)
        # pathPlotName = f"Experiments/Budget Experiments3/generated/RSEO/40k/map{rSeed}path{j + 1}"
        # msePlotName = f"Experiments/Budget Experiments3/generated/RSEO/40k/map{rSeed}mse{j + 1}"
        # outputTextName = f"Experiments/Budget Experiments3/generated/RSEO/40k/map{rSeed}output{j + 1}.txt"
        # RSEO(HP_gdf=hoverPoints, ax1=ax, df=df, fig1=fig, sensorNames=sensorNames,
        #      savePlots=True, generate=True, pathPlotName=pathPlotName, msePlotName=msePlotName,
        #      energyBudget=40000, outputTextName=outputTextName, numSensors=40, addToOriginal=True)
        # fig, ax = plt.subplots(figsize=(8, 8))
        # ax.set_xlim(-100, 1300 + 100)
        # ax.set_ylim(-100, 800 + 100)
        # hoverPoints, sensorNames = getHoverPoints(sensorsGDF, commRadius=70, ax=ax, height=15)
        # pathPlotName = f"Experiments/Budget Experiments3/generated/RSEO/45k/map{rSeed}path{j + 1}"
        # msePlotName = f"Experiments/Budget Experiments3/generated/RSEO/45k/map{rSeed}mse{j + 1}"
        # outputTextName = f"Experiments/Budget Experiments3/generated/RSEO/45k/map{rSeed}output{j + 1}.txt"
        # RSEO(HP_gdf=hoverPoints, ax1=ax, df=df, fig1=fig, sensorNames=sensorNames,
        #      savePlots=True, generate=True, pathPlotName=pathPlotName, msePlotName=msePlotName,
        #      energyBudget=45000, outputTextName=outputTextName, numSensors=40, addToOriginal=True)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(-100, 1300 + 100)
        ax.set_ylim(-100, 800 + 100)
        hoverPoints, sensorNames = getHoverPoints(sensorsGDF, commRadius=70, ax=ax, height=15)
        pathPlotName = f"Experiments/Budget Experiments3/generated/RSEO/50k/map{rSeed}path{j + 4}"
        msePlotName = f"Experiments/Budget Experiments3/generated/RSEO/50k/map{rSeed}mse{j + 4}"
        outputTextName = f"Experiments/Budget Experiments3/generated/RSEO/50k/map{rSeed}output{j + 4}.txt"
        RSEO(HP_gdf=hoverPoints, ax1=ax, df=df, fig1=fig, sensorNames=sensorNames,
             savePlots=True, generate=True, pathPlotName=pathPlotName, msePlotName=msePlotName,
             energyBudget=50000, outputTextName=outputTextName, numSensors=40, addToOriginal=True)
        # fig, ax = plt.subplots(figsize=(8, 8))
        # ax.set_xlim(-100, 1300 + 100)
        # ax.set_ylim(-100, 800 + 100)
        # hoverPoints, sensorNames = getHoverPoints(sensorsGDF, commRadius=70, ax=ax, height=15)
        # pathPlotName = f"Experiments/Budget Experiments3/generated/RSEO/55k/map{rSeed}path{j + 1}"
        # msePlotName = f"Experiments/Budget Experiments3/generated/RSEO/55k/map{rSeed}mse{j + 1}"
        # outputTextName = f"Experiments/Budget Experiments3/generated/RSEO/55k/map{rSeed}output{j + 1}.txt"
        # RSEO(HP_gdf=hoverPoints, ax1=ax, df=df, fig1=fig, sensorNames=sensorNames,
        #      savePlots=True, generate=True, pathPlotName=pathPlotName, msePlotName=msePlotName,
        #      energyBudget=55000, outputTextName=outputTextName, numSensors=40, addToOriginal=True)


synthBudgetTests()



def synthNumSensorsTests():
    for i in range(3):
        rSeed = i + 1
        random.seed(rSeed)
        pathPlotName = f"Experiments/numSensorsTests/RSEO/20s/map{i + 1}path1"
        msePlotName = f"Experiments/numSensorsTests/RSEO/20s/map{i + 1}mse1"
        outputTextName = f"Experiments/numSensorsTests/RSEO/20s/map{i + 1}output1.txt"
        RSEO(savePlots=True, generate=True, pathPlotName=pathPlotName, msePlotName=msePlotName,
             energyBudget=60000, outputTextName=outputTextName, numSensors=20)
    for i in range(3):
        rSeed = i + 1
        random.seed(rSeed)
        pathPlotName = f"Experiments/numSensorsTests/RSEO/30s/map{i + 1}path1"
        msePlotName = f"Experiments/numSensorsTests/RSEO/30s/map{i + 1}mse1"
        outputTextName = f"Experiments/numSensorsTests/RSEO/30s/map{i + 1}output1.txt"
        RSEO(savePlots=True, generate=True, pathPlotName=pathPlotName, msePlotName=msePlotName,
             energyBudget=60000, outputTextName=outputTextName, numSensors=30)
    for i in range(3):
        rSeed = i + 1
        random.seed(rSeed)
        pathPlotName = f"Experiments/numSensorsTests/RSEO/40s/map{i + 1}path1"
        msePlotName = f"Experiments/numSensorsTests/RSEO/40s/map{i + 1}mse1"
        outputTextName = f"Experiments/numSensorsTests/RSEO/40s/map{i + 1}output1.txt"
        RSEO(savePlots=True, generate=True, pathPlotName=pathPlotName, msePlotName=msePlotName,
             energyBudget=60000, outputTextName=outputTextName, numSensors=40)
    for i in range(3):
        rSeed = i + 1
        random.seed(rSeed)
        pathPlotName = f"Experiments/numSensorsTests/RSEO/50s/map{i + 1}path1"
        msePlotName = f"Experiments/numSensorsTests/RSEO/50s/map{i + 1}mse1"
        outputTextName = f"Experiments/numSensorsTests/RSEO/50s/map{i + 1}output1.txt"
        RSEO(savePlots=True, generate=True, pathPlotName=pathPlotName, msePlotName=msePlotName,
             energyBudget=60000, outputTextName=outputTextName, numSensors=50)
    for i in range(3):
        rSeed = i + 1
        random.seed(rSeed)
        pathPlotName = f"Experiments/numSensorsTests/RSEO/60s/map{i + 1}path1"
        msePlotName = f"Experiments/numSensorsTests/RSEO/60s/map{i + 1}mse1"
        outputTextName = f"Experiments/numSensorsTests/RSEO/60s/map{i + 1}output1.txt"
        RSEO(savePlots=True, generate=True, pathPlotName=pathPlotName, msePlotName=msePlotName,
             energyBudget=60000, outputTextName=outputTextName, numSensors=60)
    for i in range(3):
        rSeed = i + 1
        random.seed(rSeed)
        pathPlotName = f"Experiments/numSensorsTests/RSEO/70s/map{i + 1}path1"
        msePlotName = f"Experiments/numSensorsTests/RSEO/70s/map{i + 1}mse1"
        outputTextName = f"Experiments/numSensorsTests/RSEO/70s/map{i + 1}output1.txt"
        RSEO(savePlots=True, generate=True, pathPlotName=pathPlotName, msePlotName=msePlotName,
             energyBudget=60000, outputTextName=outputTextName, numSensors=70)
# synthNumSensorsTests()

def numSensorsTests2():
    for i in range(1):
        rSeed = i + 1
        random.seed(rSeed)
        pathPlotName = f"Experiments/numSensorsTests2/RSEO/20s/map{i+1}path1"
        msePlotName = f"Experiments/numSensorsTests2/RSEO/20s/map{i+1}mse1"
        outputTextName = f"Experiments/numSensorsTests2/RSEO/20s/map{i+1}output1.txt"
        RSEO(savePlots=True, generate=True, pathPlotName=pathPlotName, msePlotName=msePlotName,
             energyBudget=60000, outputTextName=outputTextName, numSensors=20)
    for i in range(1):
        rSeed = i + 1
        random.seed(rSeed)
        pathPlotName = f"Experiments/numSensorsTests2/RSEO/30s/map{i+1}path1"
        msePlotName = f"Experiments/numSensorsTests2/RSEO/30s/map{i+1}mse1"
        outputTextName = f"Experiments/numSensorsTests2/RSEO/30s/map{i+1}output1.txt"
        RSEO(savePlots=True, generate=True, pathPlotName=pathPlotName, msePlotName=msePlotName,
             energyBudget=60000, outputTextName=outputTextName, numSensors=30)
    for i in range(1):
        rSeed = i + 1
        random.seed(rSeed)
        pathPlotName = f"Experiments/numSensorsTests2/RSEO/40s/map{i+1}path1"
        msePlotName = f"Experiments/numSensorsTests2/RSEO/40s/map{i+1}mse1"
        outputTextName = f"Experiments/numSensorsTests2/RSEO/40s/map{i+1}output1.txt"
        RSEO(savePlots=True, generate=True, pathPlotName=pathPlotName, msePlotName=msePlotName,
             energyBudget=60000, outputTextName=outputTextName, numSensors=40)
    # for i in range(1):
    #     rSeed = i + 1
    #     random.seed(rSeed)
    #     pathPlotName = f"Experiments/numSensorsTests2/RSEO/50s/map{i+1}path1"
    #     msePlotName = f"Experiments/numSensorsTests2/RSEO/50s/map{i+1}mse1"
    #     outputTextName = f"Experiments/numSensorsTests2/RSEO/50s/map{i+1}output1.txt"
    #     RSEO(savePlots=True, generate=True, pathPlotName=pathPlotName, msePlotName=msePlotName,
    #          energyBudget=60000, outputTextName=outputTextName, numSensors=50)
    # for i in range(1):
    #     rSeed = i + 1
    #     random.seed(rSeed)
    #     pathPlotName = f"Experiments/numSensorsTests2/RSEO/60s/map{i+1}path1"
    #     msePlotName = f"Experiments/numSensorsTests2/RSEO/60s/map{i+1}mse1"
    #     outputTextName = f"Experiments/numSensorsTests2/RSEO/60s/map{i+1}output1.txt"
    #     RSEO(savePlots=True, generate=True, pathPlotName=pathPlotName, msePlotName=msePlotName,
    #          energyBudget=60000, outputTextName=outputTextName, numSensors=60)
    # for i in range(1):
    #     rSeed = i + 1
    #     random.seed(rSeed)
    #     pathPlotName = f"Experiments/numSensorsTests2/RSEO/70s/map{i+1}path1"
    #     msePlotName = f"Experiments/numSensorsTests2/RSEO/70s/map{i+1}mse1"
    #     outputTextName = f"Experiments/numSensorsTests2/RSEO/70s/map{i+1}output1.txt"
    #     RSEO(savePlots=True, generate=True, pathPlotName=pathPlotName, msePlotName=msePlotName,
    #          energyBudget=60000, outputTextName=outputTextName, numSensors=70)

def numSensorsTests3():
    for i in range(3):
        rSeed = i + 1
        random.seed(rSeed)
        pathPlotName = f"Experiments/numSensorsTests3/RSEO/20s/map{i+1}path1"
        msePlotName = f"Experiments/numSensorsTests3/RSEO/20s/map{i+1}mse1"
        outputTextName = f"Experiments/numSensorsTests3/RSEO/20s/map{i+1}output1.txt"
        RSEO(savePlots=True, generate=True, pathPlotName=pathPlotName, msePlotName=msePlotName,
             energyBudget=50000, outputTextName=outputTextName, numSensors=20, addToOriginal=True)
    for i in range(3):
        rSeed = i + 1
        random.seed(rSeed)
        pathPlotName = f"Experiments/numSensorsTests3/RSEO/30s/map{i+1}path1"
        msePlotName = f"Experiments/numSensorsTests3/RSEO/30s/map{i+1}mse1"
        outputTextName = f"Experiments/numSensorsTests3/RSEO/30s/map{i+1}output1.txt"
        RSEO(savePlots=True, generate=True, pathPlotName=pathPlotName, msePlotName=msePlotName,
             energyBudget=50000, outputTextName=outputTextName, numSensors=30, addToOriginal=True)
    for i in range(3):
        rSeed = i + 1
        random.seed(rSeed)
        pathPlotName = f"Experiments/numSensorsTests3/RSEO/40s/map{i+1}path1"
        msePlotName = f"Experiments/numSensorsTests3/RSEO/40s/map{i+1}mse1"
        outputTextName = f"Experiments/numSensorsTests3/RSEO/40s/map{i+1}output1.txt"
        RSEO(savePlots=True, generate=True, pathPlotName=pathPlotName, msePlotName=msePlotName,
             energyBudget=50000, outputTextName=outputTextName, numSensors=40, addToOriginal=True)
    for i in range(3):
        rSeed = i + 1
        random.seed(rSeed)
        pathPlotName = f"Experiments/numSensorsTests3/RSEO/50s/map{i+1}path1"
        msePlotName = f"Experiments/numSensorsTests3/RSEO/50s/map{i+1}mse1"
        outputTextName = f"Experiments/numSensorsTests3/RSEO/50s/map{i+1}output1.txt"
        RSEO(savePlots=True, generate=True, pathPlotName=pathPlotName, msePlotName=msePlotName,
             energyBudget=50000, outputTextName=outputTextName, numSensors=50, addToOriginal=True)
    for i in range(3):
        rSeed = i + 1
        random.seed(rSeed)
        pathPlotName = f"Experiments/numSensorsTests3/RSEO/60s/map{i+1}path1"
        msePlotName = f"Experiments/numSensorsTests3/RSEO/60s/map{i+1}mse1"
        outputTextName = f"Experiments/numSensorsTests3/RSEO/60s/map{i+1}output1.txt"
        RSEO(savePlots=True, generate=True, pathPlotName=pathPlotName, msePlotName=msePlotName,
             energyBudget=50000, outputTextName=outputTextName, numSensors=60, addToOriginal=True)
    for i in range(3):
        rSeed = i + 1
        random.seed(rSeed)
        pathPlotName = f"Experiments/numSensorsTests3/RSEO/70s/map{i+1}path1"
        msePlotName = f"Experiments/numSensorsTests3/RSEO/70s/map{i+1}mse1"
        outputTextName = f"Experiments/numSensorsTests3/RSEO/70s/map{i+1}output1.txt"
        RSEO(savePlots=True, generate=True, pathPlotName=pathPlotName, msePlotName=msePlotName,
             energyBudget=50000, outputTextName=outputTextName, numSensors=70, addToOriginal=True)

# numSensorsTests2()
# numSensorsTests3()


# NumSensorsTests40k2/RSEO3 try with more seeds

def numSensorsTests40k():
    for j in range(2, 3):
        rSeeds = [7]
        for rSeed in rSeeds:
            random.seed(rSeed)  # ONE OF THESE NEEDS TO GO
            np.random.seed(rSeed)
            dataFolder = 'CAF_Sensor_Dataset_2/caf_sensors/Hourly'
            df = processData(dataFolder)
            sensorsGDF, df = addSensorsUniformRandom(df=df, numSensors=80)
            df = normalizeData(df)
            df = df[np.random.permutation(df.columns)]

            croppedDF = df.iloc[:, :20]
            croppedSensorsGDF = sensorsGDF[sensorsGDF['Location'].isin(croppedDF.columns)]
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_xlim(-100, 1300 + 100)
            ax.set_ylim(-100, 800 + 100)
            hoverPoints, sensorNames = getHoverPoints(croppedSensorsGDF, commRadius=70, ax=ax, height=15)
            pathPlotName = f"Experiments/numSensorsTests40k2/RSEO3/20s/map{rSeed}path{j}"
            msePlotName = f"Experiments/numSensorsTests40k2/RSEO3/20s/map{rSeed}mse{j}"
            outputTextName = f"Experiments/numSensorsTests40k2/RSEO3/20s/map{rSeed}output{j}.txt"
            RSEO(HP_gdf=hoverPoints, ax1=ax, df=croppedDF, fig1=fig, sensorNames=sensorNames,
                 savePlots=True, generate=True, pathPlotName=pathPlotName, msePlotName=msePlotName,
                 energyBudget=40000, outputTextName=outputTextName, numSensors=20, addToOriginal=True)

            # croppedDF = df.iloc[:, :30]
            # croppedSensorsGDF = sensorsGDF[sensorsGDF['Location'].isin(croppedDF.columns)]
            # fig, ax = plt.subplots(figsize=(8, 8))
            # ax.set_xlim(-100, 1300 + 100)
            # ax.set_ylim(-100, 800 + 100)
            # hoverPoints, sensorNames = getHoverPoints(croppedSensorsGDF, commRadius=70, ax=ax, height=15)
            # pathPlotName = f"Experiments/numSensorsTests40k2/RSEO3/30s/map{rSeed}path{j}"
            # msePlotName = f"Experiments/numSensorsTests40k2/RSEO3/30s/map{rSeed}mse{j}"
            # outputTextName = f"Experiments/numSensorsTests40k2/RSEO3/30s/map{rSeed}output{j}.txt"
            # RSEO(HP_gdf=hoverPoints, ax1=ax, df=croppedDF, fig1=fig, sensorNames=sensorNames,
            #      savePlots=True, generate=True, pathPlotName=pathPlotName, msePlotName=msePlotName,
            #      energyBudget=40000, outputTextName=outputTextName, numSensors=30, addToOriginal=True)
            #
            # croppedDF = df.iloc[:, :40]
            # croppedSensorsGDF = sensorsGDF[sensorsGDF['Location'].isin(croppedDF.columns)]
            # fig, ax = plt.subplots(figsize=(8, 8))
            # ax.set_xlim(-100, 1300 + 100)
            # ax.set_ylim(-100, 800 + 100)
            # hoverPoints, sensorNames = getHoverPoints(croppedSensorsGDF, commRadius=70, ax=ax, height=15)
            # pathPlotName = f"Experiments/numSensorsTests40k2/RSEO3/40s/map{rSeed}path{j}"
            # msePlotName = f"Experiments/numSensorsTests40k2/RSEO3/40s/map{rSeed}mse{j}"
            # outputTextName = f"Experiments/numSensorsTests40k2/RSEO3/40s/map{rSeed}output{j}.txt"
            # RSEO(HP_gdf=hoverPoints, ax1=ax, df=croppedDF, fig1=fig, sensorNames=sensorNames,
            #      savePlots=True, generate=True, pathPlotName=pathPlotName, msePlotName=msePlotName,
            #      energyBudget=40000, outputTextName=outputTextName, numSensors=40, addToOriginal=True)
            #
            # croppedDF = df.iloc[:, :50]
            # croppedSensorsGDF = sensorsGDF[sensorsGDF['Location'].isin(croppedDF.columns)]
            # fig, ax = plt.subplots(figsize=(8, 8))
            # ax.set_xlim(-100, 1300 + 100)
            # ax.set_ylim(-100, 800 + 100)
            # hoverPoints, sensorNames = getHoverPoints(croppedSensorsGDF, commRadius=70, ax=ax, height=15)
            # pathPlotName = f"Experiments/numSensorsTests40k2/RSEO3/50s/map{rSeed}path{j}"
            # msePlotName = f"Experiments/numSensorsTests40k2/RSEO3/50s/map{rSeed}mse{j}"
            # outputTextName = f"Experiments/numSensorsTests40k2/RSEO3/50s/map{rSeed}output{j}.txt"
            # RSEO(HP_gdf=hoverPoints, ax1=ax, df=croppedDF, fig1=fig, sensorNames=sensorNames,
            #      savePlots=True, generate=True, pathPlotName=pathPlotName, msePlotName=msePlotName,
            #      energyBudget=40000, outputTextName=outputTextName, numSensors=50, addToOriginal=True)
            #
            # croppedDF = df.iloc[:, :60]
            # croppedSensorsGDF = sensorsGDF[sensorsGDF['Location'].isin(croppedDF.columns)]
            # fig, ax = plt.subplots(figsize=(8, 8))
            # ax.set_xlim(-100, 1300 + 100)
            # ax.set_ylim(-100, 800 + 100)
            # hoverPoints, sensorNames = getHoverPoints(croppedSensorsGDF, commRadius=70, ax=ax, height=15)
            # pathPlotName = f"Experiments/numSensorsTests40k2/RSEO3/60s/map{rSeed}path{j}"
            # msePlotName = f"Experiments/numSensorsTests40k2/RSEO3/60s/map{rSeed}mse{j}"
            # outputTextName = f"Experiments/numSensorsTests40k2/RSEO3/60s/map{rSeed}output{j}.txt"
            # RSEO(HP_gdf=hoverPoints, ax1=ax, df=croppedDF, fig1=fig, sensorNames=sensorNames,
            #      savePlots=True, generate=True, pathPlotName=pathPlotName, msePlotName=msePlotName,
            #      energyBudget=40000, outputTextName=outputTextName, numSensors=60, addToOriginal=True)
            #
            # croppedDF = df.iloc[:, :70]
            # croppedSensorsGDF = sensorsGDF[sensorsGDF['Location'].isin(croppedDF.columns)]
            # fig, ax = plt.subplots(figsize=(8, 8))
            # ax.set_xlim(-100, 1300 + 100)
            # ax.set_ylim(-100, 800 + 100)
            # hoverPoints, sensorNames = getHoverPoints(croppedSensorsGDF, commRadius=70, ax=ax, height=15)
            # pathPlotName = f"Experiments/numSensorsTests40k2/RSEO3/70s/map{rSeed}path{j}"
            # msePlotName = f"Experiments/numSensorsTests40k2/RSEO3/70s/map{rSeed}mse{j}"
            # outputTextName = f"Experiments/numSensorsTests40k2/RSEO3/70s/map{rSeed}output{j}.txt"
            # RSEO(HP_gdf=hoverPoints, ax1=ax, df=croppedDF, fig1=fig, sensorNames=sensorNames,
            #      savePlots=True, generate=True, pathPlotName=pathPlotName, msePlotName=msePlotName,
            #      energyBudget=40000, outputTextName=outputTextName, numSensors=70, addToOriginal=True)
# numSensorsTests40k()

# rSeed = 3
# random.seed(rSeed)
# pathPlotName = f"Experiments/numSensorsTests40k/RSEO/30s/map{rSeed}path1"
# msePlotName = f"Experiments/numSensorsTests40k/RSEO/30s/map{rSeed}mse1"
# outputTextName = f"Experiments/numSensorsTests40k/RSEO/30s/map{rSeed}output1.txt"
# RSEO(savePlots=True, generate=True, pathPlotName=pathPlotName, msePlotName=msePlotName,
#      energyBudget=40000, outputTextName=outputTextName, numSensors=30, addToOriginal=True)
# numSensorsTests40k()

def numSensorsTests40k3():
    # for i in range(2):
    # rSeed = 1
    # random.seed(rSeed)
    # pathPlotName = f"Experiments/numSensorsTests40k3/RSEO/70s/map{i+1}path1"
    # msePlotName = f"Experiments/numSensorsTests40k3/RSEO/70s/map{i+1}mse1"
    # outputTextName = f"Experiments/numSensorsTests40k3/RSEO/70s/map{i+1}output1.txt"
    # RSEO(savePlots=True, generate=True, pathPlotName=pathPlotName, msePlotName=msePlotName,
    #      energyBudget=40000, outputTextName=outputTextName, numSensors=70, addToOriginal=True)
    # for i in range(2):
    i = 1
    rSeed = i
    random.seed(rSeed)
    pathPlotName = f"Experiments/numSensorsTests40k3/RSEO/20s/map{i+1}path1"
    msePlotName = f"Experiments/numSensorsTests40k3/RSEO/20s/map{i+1}mse1"
    outputTextName = f"Experiments/numSensorsTests40k3/RSEO/20s/map{i+1}output1.txt"
    RSEO(savePlots=True, generate=True, pathPlotName=pathPlotName, msePlotName=msePlotName,
         energyBudget=40000, outputTextName=outputTextName, numSensors=20, addToOriginal=True)
    for i in range(2):
        rSeed = i
        random.seed(rSeed)
        pathPlotName = f"Experiments/numSensorsTests40k3/RSEO/30s/map{i+1}path1"
        msePlotName = f"Experiments/numSensorsTests40k3/RSEO/30s/map{i+1}mse1"
        outputTextName = f"Experiments/numSensorsTests40k3/RSEO/30s/map{i+1}output1.txt"
        RSEO(savePlots=True, generate=True, pathPlotName=pathPlotName, msePlotName=msePlotName,
             energyBudget=40000, outputTextName=outputTextName, numSensors=30, addToOriginal=True)
    # for i in range(2):
    # rSeed = i
    # random.seed(rSeed)
    # pathPlotName = f"Experiments/numSensorsTests40k3/RSEO/40s/map{i+1}path1"
    # msePlotName = f"Experiments/numSensorsTests40k3/RSEO/40s/map{i+1}mse1"
    # outputTextName = f"Experiments/numSensorsTests40k3/RSEO/40s/map{i+1}output1.txt"
    # RSEO(savePlots=True, generate=True, pathPlotName=pathPlotName, msePlotName=msePlotName,
    #      energyBudget=50000, outputTextName=outputTextName, numSensors=40, addToOriginal=True)
    # # for i in range(2):
    i = 0
    rSeed = i
    random.seed(rSeed)
    pathPlotName = f"Experiments/numSensorsTests40k3/RSEO/50s/map{i+1}path1"
    msePlotName = f"Experiments/numSensorsTests40k3/RSEO/50s/map{i+1}mse1"
    outputTextName = f"Experiments/numSensorsTests40k3/RSEO/50s/map{i+1}output1.txt"
    RSEO(savePlots=True, generate=True, pathPlotName=pathPlotName, msePlotName=msePlotName,
         energyBudget=40000, outputTextName=outputTextName, numSensors=50, addToOriginal=True)
    # for i in range(2):
    # rSeed = i
    # random.seed(rSeed)
    # pathPlotName = f"Experiments/numSensorsTests40k3/RSEO/60s/map{i+1}path1"
    # msePlotName = f"Experiments/numSensorsTests40k3/RSEO/60s/map{i+1}mse1"
    # outputTextName = f"Experiments/numSensorsTests40k3/RSEO/60s/map{i+1}output1.txt"
    # RSEO(savePlots=True, generate=True, pathPlotName=pathPlotName, msePlotName=msePlotName,
    #      energyBudget=40000, outputTextName=outputTextName, numSensors=60, addToOriginal=True)


# numSensorsTests40k3()

def origTechTests():
    for i in range(3, 5):
        pathPlotName = f"Experiments/techs/original/RSEO/Zig/path{i + 1}"
        msePlotName = f"Experiments/techs/original/RSEO/Zig/mse{i + 1}"
        outputTextName = f"Experiments/techs/original/RSEO/Zig/output{i + 1}.txt"
        RSEO(generate=False, savePlots=True, dataSize=10, energyBudget=40000, transferRate=.25,
             communicationRadius=50, droneHeight=9, pathPlotName=pathPlotName, msePlotName=msePlotName,
             outputTextName=outputTextName)
        pathPlotName = f"Experiments/techs/original/RSEO/WiFi/path{i + 1}"
        msePlotName = f"Experiments/techs/original/RSEO/WiFi/mse{i + 1}"
        outputTextName = f"Experiments/techs/original/RSEO/WiFi/output{i + 1}.txt"
        RSEO(generate=False, savePlots=True, dataSize=10, energyBudget=40000, transferRate=9,
             communicationRadius=70, droneHeight=9, pathPlotName=pathPlotName, msePlotName=msePlotName,
             outputTextName=outputTextName)
        pathPlotName = f"Experiments/techs/original/RSEO/BT/path{i + 1}"
        msePlotName = f"Experiments/techs/original/RSEO/BT/mse{i + 1}"
        outputTextName = f"Experiments/techs/original/RSEO/BT/output{i + 1}.txt"
        RSEO(generate=False, savePlots=True, dataSize=10, energyBudget=40000, transferRate=25,
             communicationRadius=10, droneHeight=9, pathPlotName=pathPlotName, msePlotName=msePlotName,
             outputTextName=outputTextName)
        pathPlotName = f"Experiments/techs/original/RSEO/UWB/path{i + 1}"
        msePlotName = f"Experiments/techs/original/RSEO/UWB/mse{i + 1}"
        outputTextName = f"Experiments/techs/original/RSEO/UWB/output{i + 1}.txt"
        RSEO(generate=False, savePlots=True, dataSize=10, energyBudget=40000, transferRate=110,
             communicationRadius=10, droneHeight=9, pathPlotName=pathPlotName, msePlotName=msePlotName,
             outputTextName=outputTextName)

# origTechTests()
