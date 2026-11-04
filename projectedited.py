# Analysis and Forecasting of Electricity Consumption Based on Weather Conditions
# ARIMA version: six graphs in one readable 3x2 figure.

import warnings                                                          #suppress these selected warnings to keep the terminal output clean
import zipfile                                                           #built in py module; open zip file and find csv data inside it
import urllib.request                                                    #urllib-py package, request-module; access internet and download files
from pathlib import Path                                                 # path of file, whether dataset exists, access file, specify where o/p

import pandas as pd                                                      #data manipulation(change,organize,10min->daily avg, remove duplicates) and analysis
import numpy as np                                                       #numerical calculations(ARIMA model selection, MAE,RMSE)
import matplotlib.pyplot as plt                                          #library;visuals
import seaborn as sns                                                    #library; correlation heatmap
from statsmodels.tsa.arima.model import ARIMA                            #To train different ARIMA models, select the best one using AIC, and generate forecasts.
from statsmodels.tools.sm_exceptions import ConvergenceWarning           #suppress convergence warning messages while testing different ARIMA models.

DATA_FILE = Path("energydata_complete.csv")                              #Stores the name of the CSV file containing our electricity consumption dataset.
ZIP_FILE = Path("appliances_energy.zip")                                 #Stores the name of the ZIP file in which the dataset is downloaded.
DATA_URL = "https://archive.ics.uci.edu/static/public/374/appliances+energy+prediction.zip"          
#Stores the online link from where our dataset is downloaded.

def load_data():
    if DATA_FILE.exists(): 
        print(f"Loading local dataset: {DATA_FILE.resolve()}")  
        #checks if the dataset is found locally, .resolve gives the complete path of the file instead of just name 
        data = pd.read_csv(DATA_FILE, low_memory=False)    
        #Reads a CSV file using Pandas.
    else:
        print("Dataset not found locally. Attempting download from UCI...")
        try:
            urllib.request.urlretrieve(DATA_URL, ZIP_FILE)                     #Downloads csv from url 
            with zipfile.ZipFile(ZIP_FILE, "r") as archive:                    #opens the zipfile in read only 
                csv_name = next(
                    name for name in archive.namelist()                        #lists the files inside the zipfile
                    if name.endswith("energydata_complete.csv")                #finds required csv file 
                )
                with archive.open(csv_name) as file_obj:                       #opens specifice file inside a zip archive 
                    data = pd.read_csv(file_obj, low_memory=False)             #considers full file when inferrinf column data types 
            data.to_csv(DATA_FILE, index=False)                                #saves dataset locally. 
            print(f"Dataset saved to {DATA_FILE.resolve()}")
        except Exception as exc:
            raise RuntimeError(
                "Dataset download failed. Download and extract the UCI dataset "
                "manually, then place energydata_complete.csv beside project.py."
            ) from exc    
        #id data is nt found 

    print(f"Loaded {data.shape[0]} rows and {data.shape[1]} columns.")  
    #gives dimensions o the dataset 0= rows, 1 =column 
    return data


def clean_data(data):
    data = data.copy() 
    #copies the orignal dataset
    data.columns = data.columns.str.strip() #gets all column names 
    if "date" not in data.columns or "Appliances" not in data.columns:
        raise ValueError("Expected 'date' and 'Appliances' columns were not found.")
    #checks for the essential columns if not raises a vale error 
    data["date"] = pd.to_datetime(data["date"], errors="coerce") 
    #converts values into datetime format and if value is invalide then turns it into NaT
    data = data.dropna(subset=["date"]) #removes missing values, checks only the dates column 
    data = data.sort_values("date").drop_duplicates().reset_index(drop=True)
    #arrages the dates; removes duplicate rows; resets  the indexing after removing duplicates
    required = ["Appliances", "T_out", "RH_out", "Windspeed", "T1", "RH_1"]
    #lists all the required values
    missing = [col for col in required if col not in data.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")

    # Ensure the columns used in numerical analysis are numeric.
    for col in required:
        data[col] = pd.to_numeric(data[col], errors="coerce")
      #Goes through each column in the required list, one by one.
      #Converts column values into numbers.
      #Converts values that cannot be understood as numbers into NaN (missing values).
    return data


def explore_data(data):
    print("\n--- Dataset Overview ---")
    print("Shape:", data.shape) #number of rows and columns 
    print("Missing values:", int(data.isnull().sum().sum())) 
    #isnull= identifies missing values, .sum= counts missing values in each column, .sum= adds the count 
    print("Duplicate rows:", int(data.duplicated().sum())) 
    #identifies and countes how many duplicae rows exist 
    print("\n--- Descriptive Statistics ---")
    print(data[["Appliances", "T_out", "RH_out", "Windspeed"]].describe().round(2))  
    #descirbe calculates staistical info, round(2) rounds value two decimal places 


def plot_energy(data, ax):
    #Shows how appliance electricity consumption changes over time using raw readings.
    ax.plot(data["date"], data["Appliances"], linewidth=0.55) #creates a line graph 
    ax.set_title("Appliance Energy Consumption Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Energy (Wh / 10 min)")
    ax.tick_params(axis="x", rotation=30, labelsize=8) #modifying x axis 
    ax.grid(alpha=0.2) #controls trannsparency 


def plot_daily_average(data, ax):
    #Shows the average daily appliance energy consumption to identify overall trends.
    daily = data.set_index("date")["Appliances"].resample("D").mean().dropna() 
    #data.setindex= makes date column the index; selects only appliance column which we need 
    # resample D= groups data into daily intervals: mean takes average.  
    ax.plot(daily.index, daily.values, linewidth=1.4)  
    #daily.index = Dates (X-axis); daily.values = Daily average consumption (Y-axis).
    ax.set_title("Daily Average Appliance Energy")
    ax.set_xlabel("Date")
    ax.set_ylabel("Average energy (Wh)")
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    ax.grid(alpha=0.2)


def plot_temperature_scatter(data, ax):
    #Shows the relationship between outside temperature and appliance energy consumption.
    clean = data.dropna(subset=["T_out", "Appliances"]) 
    #dropna= removes rows with missing values; subset= specifies columns to check
    ax.scatter(clean["T_out"], clean["Appliances"], alpha=0.25, s=8)
    #creates scatter plot, alpha= makes dots transparent, dot size
    ax.set_title("Outside Temperature vs Energy")
    ax.set_xlabel("Outside temperature (°C)")
    ax.set_ylabel("Energy (Wh / 10 min)")
    ax.grid(alpha=0.2)


def analyse_weather(data, ax):
    #Its purpose is to analyse the relationship between weather conditions 
    #and appliance energy consumption using correlation and a heatmap.
    columns = ["Appliances", "T_out", "RH_out", "Windspeed", "T1", "RH_1"]         
    #columns for analysis 
    corr = data[columns].corr(numeric_only=True)                                   
    #calculates correlation between selected columns 
    print("\n--- Correlation with Appliance Energy ---")
    print(corr["Appliances"].sort_values(ascending=False).round(3))
    #selects correlatation vales associated with appliance energy;
    #sort values arranges the values
    #ascending false makes it highest to lowest 
    sns.heatmap(
        corr, annot=True, cmap="coolwarm", center=0, fmt=".2f",
        ax=ax, annot_kws={"size": 8}, cbar_kws={"shrink": 0.8}
    ) 
    #creates heatmap; annot displays numerical values inside the boxes:
    #cmap= uses cool and warm colours 
    #center = 0 makes zeros center of colour scale. fmt=".2f" : displays values upto two decimal places
    #ax = ax draws heatmap in suplied region 
    #cbar_kws={"shrink": 0.8} makes colour bar smaller. 
    ax.set_title("Correlation Heatmap")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)


def select_arima_model(train):
    #AutoRegressive Integrated Moving Average
    #takes training data as input
    """
    Select a non-seasonal ARIMA(p,d,q) model using AIC.
    Only training observations are used during model selection.
    """
    best_aic = np.inf                        #Starts with infinity so that the first valid model's AIC will be smaller. Akaike information creation                         
    best_order = None                        #stores parameters, initially no combos of ARIMA 
    best_model = None                        #different combinations give different results so we need the best model 

    # A compact search keeps runtime reasonable for this student project.
    candidate_orders = [
        (p, d, q)
        #p- no. of past values; d- number of times data is differenced;
        #q- number of past forecast errors used 
        for p in range(4)
        for d in range(2)
        for q in range(4) 
        if not (p == 0 and d == 0 and q == 0)
    ]

    for order in candidate_orders:
        try:
            with warnings.catch_warnings():                           #Temporarily manages warnings.
                warnings.simplefilter("ignore", ConvergenceWarning)   #Hides convergence warnings.
                warnings.simplefilter("ignore", UserWarning)          #Hides general user warnings.
                fitted = ARIMA(
                    train,
                    order=order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                ).fit()

                #Atual model fitting occurs; 
                #arima creates model using training data; 
                #supplies order; 
                #does not force autoregressive and moving average part to satisfy invertibility restrictions 
                #Fits the model and estimates its parameters from the training data.

            if np.isfinite(fitted.aic) and fitted.aic < best_aic:  
                #checks whether the AIC is a valid finite number &
                # checks whether the current models AIC is lower than the best AIC found 
                best_aic = fitted.aic
                best_order = order
                best_model = fitted
        except Exception:
            # Some combinations may fail to converge or be unsuitable.
            continue

    if best_model is None:
        raise RuntimeError(
            "ARIMA model fitting failed for all tested orders. "
            "Check the daily data and statsmodels installation."
        )

    return best_order, best_model, best_aic


def make_forecast(data):
    daily = (
        data.set_index("date")["Appliances"]
        .resample("D").mean()
        .dropna()
        .astype(float)                            # converts values into decimal 
    )

    if len(daily) < 30:     
        #If we have fewer than 30 days of data, the program stops and displays an error.
        raise ValueError("At least 30 daily observations are recommended for ARIMA.")

    split = int(len(daily) * 0.8)
    train, test = daily.iloc[:split], daily.iloc[split:]

    print("\n--- ARIMA Model Selection ---")
    order, model, aic = select_arima_model(train)
    print(f"Selected ARIMA order: {order}")     
    print(f"Training AIC: {aic:.2f}")
    #calls the previously defined function 
    # Forecast the complete held-out test period from the training-only model.
    forecast_result = model.get_forecast(steps=len(test))  
    #function generates forecast usinb traiined arima model
    #predicts as many days as there are in the testing data and stores data 
    prediction = pd.Series(
        np.asarray(forecast_result.predicted_mean),
        index=test.index,
        name="Forecast"
    )
    #Creates a Pandas Series to store predictions.
    #Converts predicted values into a NumPy array.  
    #Retrieves the predicted energy consumption values.

    mae = float(np.mean(np.abs(test.values - prediction.values)))
    rmse = float(np.sqrt(np.mean((test.values - prediction.values) ** 2)))

    print("\n--- Forecast Evaluation ---")
    print("Training days:", len(train))
    print("Testing days:", len(test))
    print(f"MAE: {mae:.2f}")                                           
    print(f"RMSE: {rmse:.2f}")
    #MAE= Mean absolute error, RMSE= root mean square error 
    return train, test, prediction, mae, rmse, order


def plot_forecast(train, test, prediction, ax):
    #Compares training data, actual test data and ARIMA predictions to evaluate forecasting performance.
    #train is historial data, test is actual data kept to test the model,
    #prediction are the values pridected by arima model 
    ax.plot(train.index, train.values, label="Training data", linewidth=1.1)
    #train indes is dates as we know, 
    #train values is the historical energy consumption data 
    ax.plot(test.index, test.values, label="Actual test data", linewidth=1.4)
    #actual energy consumption during the dates 
    ax.plot(test.index, prediction.values, label="ARIMA forecast", linewidth=1.4)
    #predicted values are values generated by the arima model 
    ax.set_title("Daily Energy: Actual vs ARIMA Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Average energy (Wh)")
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)


def plot_forecast_errors(test, prediction, ax):
    #checks how accurate the arima model is and if it predictas energy consumption correctly 
    errors = test.values - prediction.values
    ax.plot(test.index, errors, marker="o", markersize=3, linewidth=1)
    ax.axhline(y=0, linestyle="--", linewidth=1)
    ax.set_title("ARIMA Forecast Errors")
    ax.set_xlabel("Date")
    ax.set_ylabel("Actual − forecast (Wh)")
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    ax.grid(alpha=0.2)


def main():
    data = clean_data(load_data())
    explore_data(data)
    train, test, prediction, mae, rmse, order = make_forecast(data)

    sns.set_theme(style="whitegrid", context="notebook")
    fig, axes = plt.subplots(3, 2, figsize=(16, 14), constrained_layout=True)
    #three rows two columns, constrained layoutprevents graphs from overlapping 

    plot_energy(data, axes[0, 0])
    plot_daily_average(data, axes[0, 1])
    plot_temperature_scatter(data, axes[1, 0])
    plot_forecast_errors(test, prediction, axes[1, 1])
    analyse_weather(data, axes[2, 0])
    plot_forecast(train, test, prediction, axes[2, 1])

    fig.suptitle(
        "Analysis and Forecasting of Electricity Consumption Based on Weather Conditions",
        fontsize=16, fontweight="bold"
    )  #gives main title above the graphs 
    output_image = Path("project_visualizations.png")
    fig.savefig(output_image, dpi=220, bbox_inches="tight") #saves all six graphs as an imahe 
    print(f"\nVisualization saved to: {output_image.resolve()}")
    print("Analysis complete. Close the figure window to exit.")
    plt.show()


if __name__ == "__main__":
    main()


#For example, imagine electricity consumption keeps increasing over several days.
#SES tends to produce a level-based forecast, whereas an appropriately configured 
#ARIMA model can account for patterns in previous observations.