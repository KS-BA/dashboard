#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 17 16:18:12 2021

@author: konstantinspuler
"""
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import matplotlib.ticker as ticker  # import a special package
from sklearn.model_selection import train_test_split
from sklearn import metrics
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from sklearn import  linear_model
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from scipy import stats

#to avoid future warnings:
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#___________________________________________________________________________________________
# DATA CLEANING

# Import data files

nt17_raw = pd.read_csv('IST_North_Tower_2017_Ene_Cons.csv')
nt18_raw = pd.read_csv('IST_North_Tower_2018_Ene_Cons.csv')
meteo_data_raw = pd.read_csv('IST_meteo_data_2017_2018_2019.csv')
holidays_pt_raw = pd.read_csv('holiday_17_18_19.csv')

#Transform the date into datetime object and make the date the index:

holidays_pt_raw.Date = pd.to_datetime(holidays_pt_raw.Date)

nt17_raw.rename(columns = {'Date_start': 'datetime'}, inplace = True)
nt17_raw.datetime = pd.to_datetime(nt17_raw.datetime, format='%d-%m-%Y %H:%M')
#nt17_raw['Date'] = nt17_raw.datetime.dt.date
nt17_raw = nt17_raw.set_index('datetime', drop = True)

nt18_raw.rename(columns = {'Date_start': 'datetime'}, inplace = True)
nt18_raw.datetime = pd.to_datetime(nt18_raw.datetime, format='%d-%m-%Y %H:%M')
#nt18_raw['Date'] = nt18_raw.datetime.dt.date
nt18_raw = nt18_raw.set_index('datetime', drop = True)

meteo_data_raw.rename(columns = {'yyyy-mm-dd hh:mm:ss': 'datetime'}, inplace = True) #renames the 'yyyy-mm-dd hh:mm:ss' column
meteo_data_raw.datetime = pd.to_datetime(meteo_data_raw.datetime) #transforming datetime column to datetime object
meteo_data_raw['Date'] = meteo_data_raw.datetime.dt.date #adds column than includes just the date
meteo_data_raw.Date = pd.to_datetime(meteo_data_raw.Date) #transforming the Date column to datetime object
meteo_data_raw['Holiday'] = meteo_data_raw['Date'].isin(holidays_pt_raw['Date']).astype(float) #adds Holiday column marking the holidays with a 1 
meteo_data_raw = meteo_data_raw.set_index('datetime', drop = True) #makes datetime column the index
meteo_data_raw['Week Day']=meteo_data_raw.index.dayofweek
meteo_data_raw['Hour']=meteo_data_raw.index.hour
meteo_data_raw = meteo_data_raw.resample('H').mean().dropna(how='all') #takes the hourly average of the different paramters and drops all empty columns
meteo_data_raw['Date']=meteo_data_raw.index.date

#meteo_data_clean = meteo_data_raw.iloc[:15386] #2134 data rows missing
meteo_data_clean = meteo_data_raw

nt17_meteo_clean = pd.merge(meteo_data_clean,nt17_raw, on='datetime',how = 'inner')
nt18_meteo_clean = pd.merge(meteo_data_clean,nt18_raw, on='datetime',how = 'inner')
nt17_meteo_clean.rename(columns = {'temp_C': 'Temperature [°C]', 'HR': 'Relative Humidity [%]', 'windSpeed_m/s': 'Wind Speed [m/s]', 'windGust_m/s': 'Wind Gust [m/s]', 'pres_mbar': 'Ambient Pressure [mbar]', 'solarRad_W/m2': 'Solar Radiation [W/m^2]', 'rain_mm/h': 'Precipitation [mm/h]', 'rain_day': 'Rain Day', 'Power_kW': 'Power [kW]'}, inplace = True)
nt18_meteo_clean.rename(columns = {'temp_C': 'Temperature [°C]', 'HR': 'Relative Humidity [%]', 'windSpeed_m/s': 'Wind Speed [m/s]', 'windGust_m/s': 'Wind Gust [m/s]', 'pres_mbar': 'Ambient Pressure [mbar]', 'solarRad_W/m2': 'Solar Radiation [W/m^2]', 'rain_mm/h': 'Precipitation [mm/h]', 'rain_day': 'Rain Day', 'Power_kW': 'Power [kW]'}, inplace = True)

nt17_18_meteo_clean = pd.concat([nt17_meteo_clean, nt18_meteo_clean], ignore_index=False)

#___________________________________________________________________________________________
## Data Analysis

#plt.plot(nt17_18_meteo_clean['Solar Radiation [W/m^2]'])

#fig, ax = plt.subplots() # create objects of the plot (figure and plot inside)
#fig.set_size_inches(20,10) # define figure size

#ax.xaxis.set_major_locator (ticker.MultipleLocator(60)) # define the interval between ticks on x axis 
    # Try changing (the number) to see what it does
#ax.xaxis.set_tick_params (which = 'major', pad = 5, labelrotation = 50)
    # parameters of major labels of x axis: pad = distance to the axis;
    # label rotation = angle of label text (in degrees)

#fig, ax = plt.subplots() # create objects of the plot (figure and plot inside)
#fig.set_size_inches(20,10) # define figure size

#ax.xaxis.set_major_locator (ticker.MultipleLocator(60)) # define the interval between ticks on x axis 
    # Try changing (the number) to see what it does
#ax.xaxis.set_tick_params (which = 'major', pad = 5, labelrotation = 50)
    # parameters of major labels of x axis: pad = distance to the axis;
    # label rotation = angle of label text (in degrees)

## Removing outliers: calculate Zcore
z17 = np.abs(stats.zscore(nt17_18_meteo_clean['Power [kW]']))

threshold = 3 # 3 sigma...Includes 99.7% of the data

nt17_18_meteo_clean2=nt17_18_meteo_clean[(z17 < 3)]

## Removing outliers: calculate IQR
Q1 = nt17_18_meteo_clean['Power [kW]'].quantile(0.25)
Q3 = nt17_18_meteo_clean['Power [kW]'].quantile(0.75)
IQR = Q3 - Q1


nt17_18_meteo_clean3 = nt17_18_meteo_clean[((nt17_18_meteo_clean['Power [kW]'] > (Q1 - 1.5 * IQR)) & (nt17_18_meteo_clean['Power [kW]'] < (Q3 + 1.5 * IQR)))]
#### Clean datafrom outliers from EDA

nt17_18_meteo_clean4 = nt17_18_meteo_clean[nt17_18_meteo_clean['Power [kW]'] > nt17_18_meteo_clean['Power [kW]'].quantile(0.05)]
#I set quantile(0.05), bc 0.25 cleaned too much..

# Plots:
    #Power plots (raw)
fig_17_18_P = px.scatter(nt17_18_meteo_clean, x = 'Date', y='Power [kW]', color_discrete_sequence = ["blue"], width=1600, height=500)
fig_17_P = px.scatter(nt17_meteo_clean, x = 'Date', y='Power [kW]', color_discrete_sequence = ["blue"], width=1600, height=500)
fig_18_P = px.scatter(nt18_meteo_clean, x = 'Date', y='Power [kW]', color_discrete_sequence = ["blue"], width=1600, height=500)
    #Power plot 17+18 (outliers removed)
fig_17_18_P_clean = px.scatter(nt17_18_meteo_clean4, x = 'Date', y='Power [kW]', color_discrete_sequence = ["blue"], width=1600, height=500)
    #Temperature plots
fig_17_18_Temp = px.scatter(nt17_18_meteo_clean, x = 'Date', y='Temperature [°C]', color_discrete_sequence = ["blue"], width=1600, height=500)
fig_17_Temp = px.scatter(nt17_meteo_clean, x = 'Date', y='Temperature [°C]', color_discrete_sequence = ["blue"], width=1600, height=500)
fig_18_Temp = px.scatter(nt18_meteo_clean, x = 'Date', y='Temperature [°C]', color_discrete_sequence = ["blue"], width=1600, height=500)
    #Solar radiation plots
fig_17_18_SR = px.scatter(nt17_18_meteo_clean, x = 'Date', y='Solar Radiation [W/m^2]', color_discrete_sequence = ["blue"], width=1600, height=500)
fig_17_SR = px.scatter(nt17_meteo_clean, x = 'Date', y='Solar Radiation [W/m^2]', color_discrete_sequence = ["blue"], width=1600, height=500)
fig_18_SR = px.scatter(nt18_meteo_clean, x = 'Date', y='Solar Radiation [W/m^2]', color_discrete_sequence = ["blue"], width=1600, height=500)


#___________________________________________________________________________________________
# DATA CLUSTERING


nt17_18_meteo_cluster = nt17_18_meteo_clean4.set_index('Date', drop = True)
#removing all unwanted columns
nt17_18_meteo_cluster = nt17_18_meteo_cluster.drop(columns=['Relative Humidity [%]','Wind Speed [m/s]', 'Wind Gust [m/s]', 'Ambient Pressure [mbar]', 'Solar Radiation [W/m^2]', 'Precipitation [mm/h]', 'Rain Day'])
# create kmeans object
model = KMeans(n_clusters=4).fit(nt17_18_meteo_cluster)
pred = model.labels_

Nc = range(1, 20)
kmeans = [KMeans(n_clusters=i) for i in Nc]
score = [kmeans[i].fit(nt17_18_meteo_cluster).score(nt17_18_meteo_cluster) for i in range(len(kmeans))]

nt17_18_meteo_cluster['Cluster']=pred
nt17_18_meteo_cluster['Cluster'] = nt17_18_meteo_cluster['Cluster'].astype(str)
ax1=px.scatter(nt17_18_meteo_cluster, x='Power [kW]',y='Week Day', color='Cluster', color_discrete_map = {"0":"red","1":"green","2":"blue", "3":"orange"}, width=550, height=550)
ax2=px.scatter(nt17_18_meteo_cluster, x='Power [kW]',y='Temperature [°C]', color='Cluster', color_discrete_map = {"0":"red","1":"green","2":"blue", "3":"orange"}, width=550, height=550)
ax3=px.scatter(nt17_18_meteo_cluster, x='Power [kW]',y='Hour', color='Cluster', color_discrete_map = {"0":"red","1":"green","2":"blue", "3":"orange"}, width=550, height=550)
ax4=px.scatter(nt17_18_meteo_cluster, x='Power [kW]',y='Holiday', color='Cluster', color_discrete_map = {"0":"red","1":"green","2":"blue", "3":"orange"}, width=550, height=550)

#3D plots Hour and Day of the week:
cluster0=nt17_18_meteo_cluster[pred==0]
cluster1=nt17_18_meteo_cluster[pred==1]
cluster2=nt17_18_meteo_cluster[pred==2]
cluster3=nt17_18_meteo_cluster[pred==3]
fig_3D_THP = px.scatter_3d(nt17_18_meteo_cluster, x='Temperature [°C]', y='Hour', z='Power [kW]', color='Cluster', color_discrete_map = {"0":"red","1":"green","2":"blue", "3":"orange"}, width=800, height=550)
fig_3D_WTP = px.scatter_3d(nt17_18_meteo_cluster, x='Week Day', y='Temperature [°C]', z='Power [kW]', color='Cluster', color_discrete_map = {"0":"red","1":"green","2":"blue", "3":"orange"}, width=800, height=550)

### Step 5: Identifying daily patterns

#https://towardsdatascience.com/clustering-electricity-profiles-with-k-means-42d6d0644d00
df = nt17_18_meteo_cluster
df = df.drop(columns=['Temperature [°C]','Week Day','Cluster','Holiday'])
df.rename(columns = {'Power [kW]': 'Power'}, inplace = True)

#Create a pivot table
df_pivot = df.pivot(columns='Hour')
df_pivot = df_pivot.dropna()
#df_pivot
#df_pivot.T.plot(figsize=(13,8), legend=False, color='blue', alpha=0.09)

sillhoute_scores = []
n_cluster_list = np.arange(2,10).astype(int)

X = df_pivot.values.copy()
    
# Very important to scale!
sc = MinMaxScaler()
X = sc.fit_transform(X)

for n_cluster in n_cluster_list:
    
    kmeans = KMeans(n_clusters=n_cluster)
    cluster_found = kmeans.fit_predict(X)
    sillhoute_scores.append(silhouette_score(X, kmeans.labels_))
    
#plt.plot(n_cluster_list,sillhoute_scores)

kmeans = KMeans(n_clusters=3)
cluster_found = kmeans.fit_predict(X)
cluster_found_sr = pd.Series(cluster_found, name='cluster')
df_pivot = df_pivot.set_index(cluster_found_sr, append=True )

#df_pivot
#fig, ax= plt.subplots(1,1, figsize=(18,10))
#color_list = ['blue','red','green']
#cluster_values = sorted(df_pivot.index.get_level_values('cluster').unique())
#for cluster, color in zip(cluster_values, color_list):
#    df_pivot.xs(cluster, level=1).T.plot(
#        ax=ax, legend=False, alpha=0.01, color=color, label= f'Cluster {cluster}'
#       )
#   df_pivot.xs(cluster, level=1).median().plot(
#       ax=ax, color=color, alpha=0.9, ls='--'
#    )

#ax.set_xticks(np.arange(1,25))
#ax.set_ylabel('kilowatts')
#ax.set_xlabel('hour')

#___________________________________________________________________________________________
# FEATURE SELECTION AND ENGINEERING


nt17_18_meteo_feature = nt17_18_meteo_cluster.drop(columns=['Cluster'])

nt17_18_meteo_feature['Power (-1h)'] = nt17_18_meteo_feature['Power [kW]'].shift(1) 
# adding a column with the previous hour consumption

nt17_18_meteo_feature = nt17_18_meteo_feature.dropna()

# Define input and outputs
X=nt17_18_meteo_feature.values

Y=X[:,4]
X=X[:,[0,1,2,3,5]] 

## Feature Selection

### Filter Methods 

#### kBest 

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

features=SelectKBest(k=2,score_func=f_regression) # Test different k number of features, uses f-test ANOVA
fit=features.fit(X,Y) #calculates the f_regression of the features
#print(fit.scores_)
features_results=fit.transform(X)
#print(features_results) # k=2:Power-1 and Hour k=3: Temperature and Power-1 and Hour

## Wrapper methods 
### Recursive Feature Elimination (RFE)
model=LinearRegression() # LinearRegression Model as Estimator
rfe=RFE(model,2)# using 2 features
rfe2=RFE(model,3) # using 3 features

fit=rfe.fit(X,Y)
fit2=rfe2.fit(X,Y)

## Emsemble methods 

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()
model.fit(X, Y)
#print(model.feature_importances_)

## Feature Extraction/Engineering 

# featuretools for automated feature engineering
#Log of temperature
nt17_18_meteo_feature['logtemp']=np.log(nt17_18_meteo_feature['Temperature [°C]'])
# Weekday square
nt17_18_meteo_feature['day^2']=np.square(nt17_18_meteo_feature['Week Day'])
# Heating degree.hour
nt17_18_meteo_feature['HDH']=np.maximum(0,nt17_18_meteo_feature['Temperature [°C]']-16)
#Not weekday
nt17_18_meteo_feature['type_of_day'] = 1
nt17_18_meteo_feature.loc[(nt17_18_meteo_feature['Holiday'] == 1) | (nt17_18_meteo_feature['Week Day'] >4 ), 'type_of_day'] = 0
#rounding the temperature values 
nt17_18_meteo_feature['Temperature [°C]']  = round(nt17_18_meteo_feature['Temperature [°C]'], 2)

# recurrent
X=nt17_18_meteo_feature.values
Y=X[:,4]
X=X[:,[0,1,2,3,5,6,7,8]] 

model = RandomForestRegressor()
model.fit(X, Y)

dataFeature = {'Variable':['Temp' ,'Holiday','Week Day', 'Hour' ,'Power(-1)','logtemp', 'day^2' ,'HDH']}
ResultsFeatureSelect = pd.DataFrame(dataFeature)
ResultsFeatureSelect['Feature Importance'] =(model.feature_importances_)


#___________________________________________________________________________________________
# REGRESSION


## Pre-processing 
nt17_18_meteo_reg = nt17_18_meteo_feature

# recurrent
X=nt17_18_meteo_reg.values
Y=X[:,4]
X=X[:,[3,5,6,8,9]] #Temp0, Holiday1, Weekday2, Hour3, Power-1 5, Logtemp6, day^2 7, HDH 8, type_of_day9
#I went through all the features and noted the change in results. 
#The ones left as the outputs are the ones that had the biggest impact.
#I can provide the list of impacts, if you would like to see them

### Split Data into training and test data 
#by default, it chooses randomly 75% of the data for training and 25% for testing
#I didn't modify the partitioning..
X_train, X_test, y_train, y_test = train_test_split(X,Y)

#----------------------------------------
## LINEAR REGRESSION 
regr = linear_model.LinearRegression()
regr.fit(X_train,y_train)
y_pred_LR = regr.predict(X_test)

data_lin=pd.DataFrame(y_test[1:200], columns=['Test'])
data_lin['Prediction'] = y_pred_LR[1:200]
fig_lin_reg = px.line(data_lin, color_discrete_map = {"Prediction":"blue", "Test":"red"}, width=700, height=400, labels=dict(x="Time", y="Power [kW]"))
fig_lin_scat = px.scatter(x = y_test, y = y_pred_LR, color_discrete_sequence = ["orange"], width=700, height=400, labels=dict(x="Real data", y="Regression Results"))
#Evaluate errors
MAE_LR=metrics.mean_absolute_error(y_test,y_pred_LR) 
MSE_LR=metrics.mean_squared_error(y_test,y_pred_LR)  
RMSE_LR= np.sqrt(metrics.mean_squared_error(y_test,y_pred_LR))
cvRMSE_LR=RMSE_LR/np.mean(y_test)
data = {'Test':['MAE', 'MSE', 'RMSE','cvRMSE']}
Results = pd.DataFrame(data)
Results['LR']=(MAE_LR,MSE_LR,RMSE_LR,cvRMSE_LR)

#----------------------------------------
#Random forest
from sklearn.ensemble import RandomForestRegressor
parameters = {'bootstrap': True,
              'min_samples_leaf': 3,
              'n_estimators': 200, 
              'min_samples_split': 15,
              'max_features': 'sqrt',
              'max_depth': 20,
              'max_leaf_nodes': None}
RF_model = RandomForestRegressor(**parameters)
#RF_model = RandomForestRegressor()
RF_model.fit(X_train, y_train)
y_pred_RF = RF_model.predict(X_test)

data_RF=pd.DataFrame(y_test[1:200], columns=['Test'])
data_RF['Prediction'] = y_pred_RF[1:200]
fig_RF_reg = px.line(data_RF, color_discrete_map = {"Prediction":"blue", "Test":"red"}, width=700, height=400, labels=dict(x="Time", y="Power [kW]"))
fig_RF_scat = px.scatter(x = y_test, y = y_pred_RF, color_discrete_sequence = ["orange"], width=700, height=400, labels=dict(x="Real data", y="Regression Results"))
#Evaluate error
MAE_RF=metrics.mean_absolute_error(y_test,y_pred_RF) 
MSE_RF=metrics.mean_squared_error(y_test,y_pred_RF)  
RMSE_RF= np.sqrt(metrics.mean_squared_error(y_test,y_pred_RF))
cvRMSE_RF=RMSE_RF/np.mean(y_test)
Results['RF']=(MAE_RF,MSE_RF,RMSE_RF,cvRMSE_RF)

#----------------------------------------
#Gradient Boosting 
from sklearn.ensemble import GradientBoostingRegressor

GB_model = GradientBoostingRegressor()
GB_model.fit(X_train, y_train)
y_pred_GB =GB_model.predict(X_test)

data_GB=pd.DataFrame(y_test[1:200], columns=['Test'])
data_GB['Prediction'] = y_pred_GB[1:200]
fig_GB = px.line(data_GB, color_discrete_map = {"Prediction":"blue", "Test":"red"}, width=700, height=400, labels=dict(x="Time", y="Power [kW]"))
fig_GB_scat = px.scatter(x = y_test, y = y_pred_GB, color_discrete_sequence = ["orange"], width=700, height=400, labels=dict(x="Real data", y="Regression Results"))

MAE_GB=metrics.mean_absolute_error(y_test,y_pred_GB) 
MSE_GB=metrics.mean_squared_error(y_test,y_pred_GB)  
RMSE_GB= np.sqrt(metrics.mean_squared_error(y_test,y_pred_GB))
cvRMSE_GB=RMSE_GB/np.mean(y_test)
Results['GB']=(MAE_GB,MSE_GB,RMSE_GB,cvRMSE_GB)


table_LR = go.Figure(data=[go.Table(header=dict(values=['MAE', 'MSE', 'RMSE','cvRMSE'],line_color='gray',
                fill_color='gray',),
                 cells=dict(values= Results['LR'],line_color='blue',
               fill_color='lightblue',
))
                     ])

table_RF = go.Figure(data=[go.Table(header=dict(values=['MAE', 'MSE', 'RMSE','cvRMSE'],line_color='gray',
                fill_color='gray',),
                 cells=dict(values= Results['RF'],line_color='blue',
               fill_color='lightblue',
))
                     ])

table_GB = go.Figure(data=[go.Table(header=dict(values=['MAE', 'MSE', 'RMSE','cvRMSE'],line_color='gray',
                fill_color='gray',),
                 cells=dict(values= Results['GB'],line_color='blue',
               fill_color='lightblue',
))
                     ])

#-__________________________________________________________________________________

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.Img(src=app.get_asset_url('IST_logo.png')),
    html.H2('POWER CONSUMPTION FORECAST - NORTH TOWER - IST'),
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='Data', value='tab-1'),
        dcc.Tab(label='Clean Power Data', value='tab-2'),
        dcc.Tab(label='Clustering', value='tab-3'),
        dcc.Tab(label='Feature Selection', value='tab-4'),
        dcc.Tab(label='Regression', value='tab-5'),
        dcc.Tab(label='Results', value='tab-6'),
        
    ]),
    html.Div(id='tabs-content')
])

def generate_table(dataframe, max_rows=10):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])

@app.callback(Output('tabs-content', 'children'),
              Input('tabs', 'value'))
             

def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            html.H3('Power, Temperature and Solar Radiation 2017 & 2018'),
            dcc.Checklist(        
        id='radio_year',
        options=[
            {'label': '2017', 'value': 2017},
            {'label': '2018', 'value': 2018},
            
        ],

        value=[2017,2018]
        
        ),
        
            dcc.Dropdown( 
        id='dropdown_measurement',
        options=[
            {'label': 'Power [kW]', 'value': 1},
            {'label': 'Temperature [°C]', 'value': 2},
            {'label': 'Solar Radiation [W/m^2]', 'value': 3},
        ], 
        value=1
        ),   
            
        html.Div(id='graphyear_png'),
        
                    ])
    
    elif tab == 'tab-2':
       return html.H3('Cleaned Power Data'), html.Div([dcc.Graph(figure=fig_17_18_P_clean),])
    
    elif tab == 'tab-3':
        return html.Div([
                html.H3('Clusters in the Data'),
                html.Div([
                    html.H4('2D-Clusters'),
                    dcc.Dropdown(
                    id='2D_Cluster',
                    options=[
                        {'label': 'Hour', 'value': 1},
                        {'label': 'Temperature [°C]', 'value': 2},
                        {'label': 'Week Day', 'value': 3},
                        {'label': 'Holiday', 'value': 4},
                        ],
                    value= 1 ),
                    html.Div(id='graph_cluster_2D')
                ], className="four columns"),
                
                html.Div([
                    html.H4('3D-Clusters'),
                    dcc.Dropdown(
                    id='3D_Cluster',
                    options=[
                        {'label': 'Temperature [°C] and Hour', 'value': 1},
                        {'label': 'Weekday and Temperature [°C]', 'value': 2},
                        ],
                    value= 1 ),
                    html.Div(id='graph_cluster_3D')
                ],className="four columns")
        ], className="row")
    
    elif tab == 'tab-4':
        return html.Div([
            html.Div([
            html.H3('Table of importance'),
                 generate_table(ResultsFeatureSelect),
        ], className="four columns"),

        html.Div([
            html.H3('Graph'),
            html.Div([html.Img(src=app.get_asset_url('FS.png'), width=1000),])
    
        ], className="eight columns"),
    ], className="row")
    
    elif tab == 'tab-5':
        return html.Div([
            html.H3('Results of different Regression methods'),
            dcc.RadioItems(
                id='Regression_method',
                options=[
                    {'label': 'Linear Regression', 'value': 1},
                    {'label': 'Gradient boosting', 'value': 2},
                    {'label': 'Random Forest', 'value': 3}
                ], 
                value=1),
                
        html.Div([
            html.H4('Prediction & Test'),
                html.Div(id='graph_regression'),  
        ], 
        style={'width': '48%', 'display': 'inline-block'}),
        html.Div([
            html.H4('Regression Results vs. Real Data'),
                html.Div(id='graph_regression_scatter'),  
        ],
        style={'width': '48%', 'display': 'inline-block'}),
        html.Div([
            html.H4('Errors'),
            html.Div(id='table_Errors'),    
            ])
    
        ])
    
    elif tab == 'tab-6':
        return html.Div([
            generate_table(Results),

    ])

@app.callback(Output('graphyear_png', 'children'), 
              Input('radio_year', 'value'),
              Input('dropdown_measurement', 'value'))

def render_figure_png(radio_year, dropdown_measurement): 
    
    if radio_year == [2017]:
        if dropdown_measurement == 1: 
            return  html.Div([dcc.Graph(figure=fig_17_P),])
        elif dropdown_measurement == 2:
            return  html.Div([dcc.Graph(figure=fig_17_Temp),])
        elif dropdown_measurement == 3:
            return  html.Div([dcc.Graph(figure=fig_17_SR),])
    elif radio_year == [2018]:
        if dropdown_measurement == 1: 
            return  html.Div([dcc.Graph(figure=fig_18_P),])
        elif dropdown_measurement == 2:
            return  html.Div([dcc.Graph(figure=fig_18_Temp),])
        elif dropdown_measurement == 3:
            return  html.Div([dcc.Graph(figure=fig_18_SR),]) 
    elif radio_year == [2017,2018]:
        if dropdown_measurement == 1:
            return  html.Div([dcc.Graph(figure=fig_17_18_P),])
        elif dropdown_measurement == 2:
            return  html.Div([dcc.Graph(figure=fig_17_18_Temp),])
        elif dropdown_measurement == 3:
            return  html.Div([dcc.Graph(figure=fig_17_18_SR),])
    elif radio_year == [2018,2017]:
        if dropdown_measurement == 1:
            return  html.Div([dcc.Graph(figure=fig_17_18_P),])
        elif dropdown_measurement == 2:
            return  html.Div([dcc.Graph(figure=fig_17_18_Temp),])
        elif dropdown_measurement == 3:
            return  html.Div([dcc.Graph(figure=fig_17_18_SR),])
    elif radio_year == []:
        return html.Div([html.Img(src='assets/NO_OPTION_CHOSEN.png', width=1600)])
        
@app.callback(Output('graph_cluster_2D', 'children'), 
              Input('2D_Cluster', 'value'))

def figure_cluster_2D(Cluster2D): 
    if Cluster2D == 1:
        return  html.Div([dcc.Graph(figure=ax3),])
    if Cluster2D == 2:
        return  html.Div([dcc.Graph(figure=ax2),])
    if Cluster2D == 3:
        return  html.Div([dcc.Graph(figure=ax1),])
    if Cluster2D == 4:
        return  html.Div([dcc.Graph(figure=ax4),])

@app.callback(Output('graph_cluster_3D', 'children'), 
              Input('3D_Cluster', 'value'))

def figure_cluster_3D(Cluster3D):
    if Cluster3D == 1:
        return html.Div([dcc.Graph(figure=fig_3D_THP),])
    if Cluster3D == 2:
        return html.Div([dcc.Graph(figure=fig_3D_WTP),])
    
@app.callback(Output('graph_regression', 'children'), 
              Input('Regression_method', 'value'))

def figure_regression(Regression_method):
    if Regression_method == 1:
        return html.Div([dcc.Graph(figure=fig_lin_reg),])
    if Regression_method == 2:
        return html.Div([dcc.Graph(figure=fig_GB),])
    if Regression_method == 3:
        return html.Div([dcc.Graph(figure=fig_RF_reg),])
    
@app.callback(Output('graph_regression_scatter', 'children'), 
              Input('Regression_method', 'value'))
def scatter_regression(Regression_method):
    if Regression_method == 1:
        return html.Div([dcc.Graph(figure=fig_lin_scat),])
    if Regression_method == 2:
        return html.Div([dcc.Graph(figure=fig_GB_scat),])
    if Regression_method == 3:
        return html.Div([dcc.Graph(figure=fig_RF_scat),])

@app.callback(Output('table_Errors', 'children'), 
              Input('Regression_method', 'value'))
def table_Regression_error(Regression_method):
    if Regression_method == 1:
        return html.Div([dcc.Graph(figure=table_LR),])
    if Regression_method == 2:
        return html.Div([dcc.Graph(figure=table_GB),])
    if Regression_method == 3:
        return html.Div([dcc.Graph(figure=table_RF),])

if __name__ == '__main__':
    app.run_server(debug=True)
