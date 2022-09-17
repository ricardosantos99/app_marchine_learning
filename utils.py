#/usr/bin/env python3
#-*- coding? utf-8 -*-
# --------------------------------------------------------------------------
# Created by: Ricardo Santos
# Creader Date: 15/09/2022
# Version 1.0
# --------------------------------------------------------------------------


import joblib 
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn import metrics
import shap


def pipeline_predict(df, obj):

    if obj == 'Predição':

        # Pipeline de transformação das variáveis 
        df['voice_mail_plan'] = df['voice_mail_plan'].astype('category')
        df['voice_mail_plan'] = df['voice_mail_plan'].cat.codes
        df['international_plan'] = df['international_plan'].astype('category')
        df['international_plan'] = df['international_plan'].cat.codes
        df = pd.get_dummies(df, columns=['area_code'])
        df['state'] = df['state'].astype('category')
        df['state'] = df['state'].cat.codes

        # Carrega o modelo treinado
        model_trained = joblib.load('rf_model_churn.pkl')

        prob = model_trained.predict_proba(df)[:,1]
        pred = model_trained.predict(df)

        return pred, prob
    
    elif obj == 'Análise Exploratória' and df == '':

        df = pd.read_csv('churn_train.csv')

        df.churn.value_counts().plot(kind='pie')
        st.pyplot()
        df.international_plan.value_counts().plot(kind='pie')
        st.pyplot()
        churn_y = df.loc[df.churn == 'yes', 'churn'].count()
        churn_n = df.loc[df.churn == 'no', 'churn'].count()
        churn_total = churn_y+churn_n
        st.write('** Distribuição de churn na base de treinamento **')
        st.write('Quantidade total de clientes ' +str(churn_total))
        st.write('Quantidade de clientes com churn ' +str(churn_y) + ' que representa ' + str(round(100*churn_y/churn_total,0)) + '% da base de clientes')
        st.write('Quantidade de clientes sem churn ' +str(churn_n) + ' que representa ' + str(round(100*churn_n/churn_total,0)) + '% da base de clientes')

        st.write('** Distribuição de churn e não churn por estado **')
        fig, ax = plt.subplots(figsize=(20,5))
        sns.countplot(data = df, x='state', order=df['state'].value_counts().index, palette='viridis', hue='churn')
        plt.xticks(rotation=90)
        plt.xlabel('State', fontsize=10, fontweight='bold')
        plt.ylabel('Customers', fontsize=10, fontweight='bold')
        plt.title('Estados dos clientes com e sem churn', fontsize=12, fontweight='bold')
        st.pyplot()

        st.write('** Matriz de correlação entre variáveis **')
        corr = df.corr()
        fig2, ax = plt.subplots(figsize=(15,7))
        sns.heatmap(corr, 
                    xticklabels=corr.columns.values,
                    yticklabels=corr.columns.values,
                    annot=True, cmap='YlGnBu', annot_kws={'size': 12}, fmt='.2f')
        st.pyplot()

    elif obj == 'Explicabilidade' and df == '':

        df = pd.read_csv('churn_train.csv')
        df['churn'] = df['churn'].astype('category')
        df['churn'] = df['churn'].cat.codes
        df['voice_mail_plan'] = df['voice_mail_plan'].astype('category')
        df['voice_mail_plan'] = df['voice_mail_plan'].cat.codes
        df['international_plan'] = df['international_plan'].astype('category')
        df['international_plan'] = df['international_plan'].cat.codes
        df['area_code'] = df['area_code'].astype('category')
        df['area_code'] = df['area_code'].cat.codes
        df['state'] = df['state'].astype('category')
        df['state'] = df['state'].cat.codes

        X = df.drop('churn', axis=1)
        y = df['churn']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7565) #Reduzir teste para 10% por conta da demora do shap values em teste

        model_trained = joblib.load('rf_model_churn.pkl')
        prob = model_trained.predict_proba(X_test)[:,1]
        pred = model_trained.predict(X_test)
        
        st.write('** Principais Métricas do modelo **')
        st.write('** AUC: **'+str(metrics.roc_auc_score(y_test,prob)))
        st.write('** Accuracy: **'+str(metrics.accuracy_score(y_test,pred)))
        st.write('** Recall: **'+str(metrics.recall_score(y_test,pred)))
        st.write('** F1-Score: **'+str(metrics.f1_score(y_test,pred)))

        st.title('Features mais importantes')
        feature_important = model_trained.get_booster().get_score(importance_type='weight')
        keys = list(feature_important.keys())
        values = list(feature_important.values())
        data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=True)
        data.nlargest(40, columns="score").plot(kind='barh', figsize = (20,10))
        st.pyplot()

        st.title('Explicabilidade usando Shap values dos primeiros três registros de teste')
        explainer = shap.Explainer(model_trained)
        shap_values = explainer(X_test)

        shap.plots.waterfall(shap_values[0])
        st.pyplot()
        shap.plots.waterfall(shap_values[1])
        st.pyplot()
        shap.plots.waterfall(shap_values[2])
        st.pyplot()

        shap.summary_plot(shap_values)
        st.pyplot()

        shap.plots.force(shap_values[3])
        st.pyplot()
        shap.plots.force(shap_values[4])
        st.pyplot()
        shap.plots.force(shap_values[5])
        st.pyplot()