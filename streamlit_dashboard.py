# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 14:57:06 2023

@author: PC.054
"""

import pandas as pd 
import numpy as np 
import streamlit as st 
import seaborn as sns 
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import feyn


import pickle

#df=pd.read_csv("Housing.csv")

st.sidebar.title("Sommaire")

pages = ["Contexte du projet", "Exploration des données", "Analyse de données", "Modélisation"]

page = st.sidebar.radio("Aller vers la page :", pages)

if page == pages[0] : 
    
    st.write("### Contexte du projet")
    
    st.write("Des relevés minutieux ont été effectués par les agents de la ville de Seattle en 2016. Néanmoins, ces relevés sont coûteux à obtenir, et à partir de ceux déjà réalisés, vous allons tenter de prédire les émissions de CO2 de bâtiments non destinés à l’habitation pour lesquels elles n’ont pas encore été mesurées.")
    
    st.write("Nous avons à notre disposition le fichier 2016_Building_Energy_Benchmarking.csv qui contient des données des différents buldings de la ville. Chaque observation en ligne correspond à un bâtiment. Chaque variable en colonne est une caractéristique du bâtiment.")
    
    st.write("Dans un premier temps, nous explorerons ce dataset. Puis nous l'analyserons visuellement pour en extraire des informations selon certains axes d'étude. Finalement nous implémenterons des modèles de Machine Learning pour prédire les émissions de CO2.")

    file_image = r'./data/image/Seattle.JPG'
    st.image(file_image)
    
if page == pages[1] :
    st.write("### Exploration des données")
    
    path_import     = "./data/source/"
    filename_import = "2016_Building_Energy_Benchmarking.csv"
    
    df = pd.read_csv(path_import + filename_import)
    
    st.write("Echantillon du dataframe :")
    st.dataframe(df.sample(5))
    
    st.write("Description du dataframe :")
    st.dataframe(df.describe(),)
    

if page == pages[2] :
    path_import     = "data/cleaned/"
    filename_import = "df_for_modelisation.csv"
    
    df = pd.read_csv(path_import + filename_import)

    
    tab1, tab2 = st.tabs(['Nbre de bâtiments par quartier et par type', 'Bâtiments par longitude et latitude'])
    with tab1:
        fig = plt.figure(figsize=(10, 4))
        sns.countplot(x='Neighborhood', data = df, order= df['Neighborhood'].value_counts().index)
        plt.title('Nbre de Batiments par quartier')
        plt.xticks(rotation = 90)
        st.pyplot(fig)  

        fig = plt.figure(figsize=(10, 4))
        sns.countplot(x='Neighborhood', data = df, order= df['Neighborhood'].value_counts().index, hue= "BuildingType")
        plt.title('Nbre de Batiments par quartier et par type de batiment')
        plt.xticks(rotation = 90)
        st.pyplot(fig) #, use_container_width=True)

    with tab2:
        fig = px.scatter(df[df['TotalGHGEmissions']>0], x="Longitude", y="Latitude", title="Bâtiments par longitude et latitude", size= "TotalGHGEmissions", color="BuildingType")
        st.plotly_chart(fig, use_container_width=True)

 
    tab3, tab4 = st.tabs(["Emission de CO2 en fonction du type de building", "Evolution des émissions de CO2 en fonction de la surface"])
    with tab3:
        fig2 = px.scatter(df, x="TotalGHGEmissions", y="BuildingType", title="Emission de CO2 en fonction du type de building")
        st.plotly_chart(fig2)
    with tab4:
        fig3 = px.scatter(df, x="PropertyGFABuilding(s)", y="TotalGHGEmissions", title="Evolution des émissions de CO2 en fonction de la surface")
        fig3.update_xaxes(title_text="Surface en m²", type="log")
        fig3.update_yaxes(title_text="Emissions de CO2", type="log") #, row=1, col=2)
        st.plotly_chart(fig3)

    variables_heatmap = ['Latitude','Longitude','NumberofBuildings','NumberofFloors',
                     'PropertyGFAParking','PropertyGFABuilding(s)','SteamUse(kBtu)',
                     'Electricity(kBtu)','NaturalGas(kBtu)','bulding_age','SiteEnergyUse(kBtu)','TotalGHGEmissions',
                     'Steamuse_bool','NaturalGas_bool']

    fig4,ax = plt.subplots()
    sns.heatmap(df[variables_heatmap].corr(), ax=ax, annot = True, annot_kws={"size": 8}, fmt=".0%", cmap="Blues")
    plt.title("Matrice de corrélation des variables")
    st.write(fig4, theme="streamlit")

    
if page == pages[3] :    
    
    filename_import = r'./data/cleaned/df_for_modelisation.csv'
    df = pd.read_csv(filename_import)
    
    
    #Kernel Ridge
    variables = ['BuildingType','PrimaryPropertyType','Latitude','Longitude','NumberofBuildings','NumberofFloors',
                     'PropertyGFAParking','PropertyGFABuilding(s)','bulding_age','Steamuse_bool',
                 'NaturalGas_bool']
    
    # Médiane de la variable 'TotalGHGEmissions' pour chaque 'PrimaryPropertyType'
    df['PrimaryPropertyType'] = df.groupby('PrimaryPropertyType')['TotalGHGEmissions'].transform('median')
    
    X = df[variables]
    y = df['SiteEnergyUse(kBtu)']

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y, 
                                                    test_size=0.3,
                                                    random_state=42)

    # loading Kernel ridge fitted model
    filename = r'./model//kernel_ridge_model.pkl'
    kernel_ridge_model = pickle.load(open(filename, 'rb'))

    train_score = r2_score(y_train, kernel_ridge_model.predict(X_train))
    test_score = r2_score(y_test, kernel_ridge_model.predict(X_test))
    
    #Création tableau résultat
    tab_resultats = pd.DataFrame(columns=['Model','R2 train','R2 test',
                                         'RMSE train','RMSE test', 'MAE train', 'MAE test'])

    tab_resultats.loc[0] = [ "Kernel Ridge",
                            r2_score(y_train,kernel_ridge_model.predict(X_train)),
                            r2_score(y_test,kernel_ridge_model.predict(X_test)),
                            mean_squared_error(y_train,kernel_ridge_model.predict(X_train), squared=False).round(0),
                            mean_squared_error(y_test,kernel_ridge_model.predict(X_test), squared=False).round(0),
                            mean_absolute_error(y_train,kernel_ridge_model.predict(X_train)).round(0),
                            mean_absolute_error(y_test,kernel_ridge_model.predict(X_test)).round(0),
                            ]
    
   # Qlattice
    df = pd.read_csv(filename_import)    
    variables2 = ['BuildingType','PrimaryPropertyType','Latitude','Longitude','NumberofBuildings','NumberofFloors',
                     'PropertyGFAParking','PropertyGFABuilding(s)','bulding_age','Steamuse_bool',
                 'NaturalGas_bool','SiteEnergyUse(kBtu)']
    
    X_qlattice = df[variables2]    
    train, test = feyn.tools.split(X_qlattice, ratio=(0.7,0.3), random_state=42)
    
    qlattice_model = feyn.Model.load('model/qlattice_model.json')
    
    train_score_Q = qlattice_model.r2_score(train)
    test_score_Q = qlattice_model.r2_score(test)

    tab_resultats.loc[1] = [ "Qlattice",
                            qlattice_model.r2_score(train),
                            qlattice_model.r2_score(test),
                            qlattice_model.rmse(train).round(0),
                            qlattice_model.rmse(test).round(0),
                            qlattice_model.mae(train).round(0),
                            qlattice_model.mae(test).round(0)
                            ]

    st.write("Résultats des 2 modèles :")
    st.dataframe(tab_resultats)

    kernel_dataset= pd.DataFrame()
    kernel_dataset['true_values']= y_test
    kernel_dataset['predict_values']= kernel_ridge_model.predict(X_test)
    kernel_dataset['model']= "kernel"
    kernel_dataset = kernel_dataset.reset_index()
    
    qlattice_dataset= pd.DataFrame()
    qlattice_dataset['true_values']= test['SiteEnergyUse(kBtu)']
    qlattice_dataset['predict_values']= qlattice_model.predict(test)
    qlattice_dataset['model']= "qlattice"
    qlattice_dataset = qlattice_dataset.reset_index()
    model_dataset = pd.concat([kernel_dataset, qlattice_dataset])

    tab1, tab2, tab3 = st.tabs(["Résultats Kernel Ridge", "Résultats Qlattice", "Kernel Vs Qlattice"])
    with tab1:
        st.dataframe(kernel_dataset)
        st.dataframe(qlattice_dataset)
        st.dataframe(model_dataset)

        fig3 = px.scatter(x= 'y_test', y= kernel_ridge_model.predict(X_test), trendline="ols", trendline_color_override="red",title= "Kernel results")
        st.plotly_chart(fig3, use_container_width=True)


        
        fig1 = px.scatter(kernel_dataset, x= 'true_values', y= 'predict_values', trendline="ols", trendline_color_override="red",title= "Kernel results values")
        st.plotly_chart(fig1, use_container_width=True)
        
        fig2 = px.scatter(kernel_dataset, x= 'true_values', y= 'predict_values', 
                          log_x=True, log_y= True,
                          trendline="ols", 
                          trendline_color_override="red",
                          title= "Kernel results Log values")
        st.plotly_chart(fig2, use_container_width=True)
        
 
    with tab2:
        fig1 = px.scatter(qlattice_dataset, x= 'true_values', y= 'predict_values',
                          trendline="ols",
                          trendline_color_override="red",
                          title= "Qlattice results values")
        
        st.plotly_chart(fig1, use_container_width=True)
        
        fig2 = px.scatter(qlattice_dataset, x= 'true_values', y= 'predict_values', 
                          log_x=True, log_y= True,
                          trendline="ols",
                          trendline_color_override="red",
                          title= "Qlattice results Log values")
        st.plotly_chart(fig2, use_container_width=True)
               
    with tab3:
        
        fig1 = px.scatter(model_dataset, x= 'true_values' , y= 'predict_values',
                          color = 'model',
                          trendline="ols", #trendline_options=dict(log_x=True),                          
                          title= "Kernel vs Qlattice results values")
        st.plotly_chart(fig1, use_container_width=True)
        
        fig2 = px.scatter(model_dataset, x= 'true_values' , y= 'predict_values',
                          color = 'model',
                          log_x=True, log_y= True,
                          trendline="ols", #trendline_options=dict(log_x=True),                          
                          title= "Kernel vs Qlattice results values")
        st.plotly_chart(fig2, use_container_width=True)
    
    
          
