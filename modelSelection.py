import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler


#iterate through folder, make new data frame indexed by date and time, columns are sensors, and value is VW_30
#from january-april 2015
def processData(folder_path):
    # Create an empty DataFrame
    df = pd.DataFrame()
    dfs = []
    # Iterate over the files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_csv(file_path, delimiter="\t")
            dfs.append(df)

    combined_df = pd.concat(dfs)

    # Pivot the DataFrame using pivot_table()
    pivoted_df = combined_df.pivot_table(index=['Date', 'Time'], columns='Location', values='VW_30cm', aggfunc='first')
    pivoted_df.reset_index(inplace=True)
    # Sort the DataFrame by date and time, only jan to april 2015
    pivoted_df['DateTime'] = pd.to_datetime(pivoted_df['Date'] + ' ' + pivoted_df['Time'])
    pivoted_df.set_index(['DateTime'], inplace=True)
    pivoted_df.sort_values('DateTime', inplace=True)
    pivoted_df.drop('Date', axis=1, inplace=True)
    pivoted_df.drop('Time', axis=1, inplace=True)

    start_date = '2015-01-01'
    end_date = '2015-04-30'
    filtered_df = pivoted_df.loc[(pivoted_df.index >= start_date) & (pivoted_df.index <= end_date)]
    #remove columns that are >95% NaN
    nan_percentage = filtered_df.isna().sum() / len(filtered_df) * 100
    threshold = 95
    columns_to_drop = nan_percentage[nan_percentage > threshold].index
    main_df = filtered_df.drop(columns=columns_to_drop)

    # Remove rows with at least one NaN value
    main_df = main_df.dropna()
    #main_df.to_csv('mainDataFrame.txt') save data
    return main_df


#find the mean squared error of Gradient Boosting, Linear Regression, Random Forest, and Support Vector
def testModels(dataFrame):
    #split sensors into 5 groups
    sensors = dataFrame.columns.tolist()
    numSensors = len(sensors)
    numGroups = 5 #also the number of folds
    kf = KFold(n_splits=numGroups)
    groups = []

    for _, groupIndex in kf.split(sensors):
        group = [sensors[i] for i in groupIndex]
        groups.append(group)

    models = [
        MultiOutputRegressor(GradientBoostingRegressor()),
        LinearRegression(),
        RandomForestRegressor(n_estimators=100, random_state=42),
        MultiOutputRegressor(SVR(kernel='linear'))
    ]
    modelMSEValues = []
    for model in models:
        groupMSEValues = []
        #Every group is the target once, and the others are the features
        for group in groups:
            targetSensors = group
            featureSensors = [sensor for sensor in sensors if sensor not in targetSensors]
            x = dataFrame[featureSensors]
            y = dataFrame[targetSensors]

            foldMSEValues = []

            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', TransformedTargetRegressor(regressor=model, transformer=StandardScaler()))
            ])
            pipeline.fit(x_train, y_train)
            predictions = pipeline.predict(x_test)
            #Calculate mean squared error of one model with one particular group as target, and one particular fold as testing data
            groupMSE = mean_squared_error(y_test, predictions)
            groupMSEValues.append(groupMSE)
        avgModelMSE = np.mean(groupMSEValues)
        modelMSEValues.append(avgModelMSE)
    print('Average MSE of Gradient Boosting Regressor: ', modelMSEValues[0])
    print('Average MSE of Linear Regression: ', modelMSEValues[1])
    print('Average MSE of Random Forest Regressor: ', modelMSEValues[2])
    print('Average MSE of Support Vector Regressor: ', modelMSEValues[3])

def selectModel():
    dataFolder = 'CAF_Sensor_Dataset_2/caf_sensors/Hourly'
    df = processData(dataFolder)
    testModels(df)

selectModel()
