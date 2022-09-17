#/usr/bin/env python3
#-*- coding? utf-8 -*-
# --------------------------------------------------------------------------
# Created by: Ricardo Santos
# Creader Date: 15/09/2022
# Version 1.0
# --------------------------------------------------------------------------

# Import das bibliotecas 

import streamlit as st 
import pandas as pd 
from PIL import Image 
from sklearn.model_selection import train_test_split
from sklearn import metrics 
import shap
import matplotlib.pyplot as plt 
import seaborn as sns 
from utils import *


st.set_option('deprecation.showPyplotGlobalUse', False)

image = Image.open('uninove.png') # Carrega imagem da Uninove
st.image(image) # Apresenta a imagem 

# Adiciona um retangulo azul com escrita em branco
html_temp = """
<div style ="background-color:blue;padding:13px">
<h1 style ="color:white;text-align:center;">Modelo de predição de Churn de clientes de telefonia móvel</h1>
</div>
"""

st.markdown(html_temp, unsafe_allow_html = True)
st.subheader('** Modelo de predição de churn usando Random Forest **')
st.markdown('** Este modelo foi treinado usando dados historicos de clientes com e sem churn **')


def main():

    st.subheader('** Selecione uma das opções abaixo: **')
    options = st.radio('O que deseja fazer? ', ('', 'Análise Exploratória', 'Predição', 'Explicabilidade'))

    if options == 'Análise Exploratória':
        pipeline_predict('', 'Análise Exploratória')

    if options == 'Predição':
        st.subheader('Insira os dados abaixo:')
        state=st.selectbox('Escolha a sigla do estado :', ['','AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA', 'HI', 'IA','ID',\
		'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME', 'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM', 'NV',\
		'NY', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VA', 'VT', 'WA', 'WI', 'WV','WY'] )
        account_length=st.number_input('Selecione o tempo como cliente :', min_value=0, max_value=250, value=0)
        area_code=st.selectbox('Selecione o codigo de area :', ['','area_code_408', 'area_code_415', 'area_code_510'])
        international_plan=st.selectbox('Selecione se o cliente tem plano internacional :', ['', 'yes', 'no'])
        voice_mail_plan=st.selectbox('Selecione se o cliente tem plano de caixa postal :',  ['', 'yes', 'no'])
        number_vmail_messages=st.slider('Insira o numero de mensagens na caixa postal', min_value=0, max_value=250, value=0)
        total_day_minutes=st.slider('Insira o numero de minutos por dia :', min_value=0, max_value=250, value=0)
        total_day_calls=st.slider('Insira o numero de ligações por dia :', min_value=0, max_value=250, value=0)
        total_day_charge=st.slider('Insira o numero de recargas por dia :', min_value=0, max_value=250, value=0)
        total_eve_minutes=st.slider('Insira o numero de minutos por tarde :', min_value=0, max_value=250, value=0)
        total_eve_calls=st.slider('Insira o numero de ligações por tarde :', min_value=0, max_value=250, value=0)
        total_eve_charge=st.slider('Insira o numero de recargas por tarde :', min_value=0, max_value=250, value=0)
        total_night_minutes=st.slider('Insira o numero de minutos por noite :', min_value=0, max_value=250, value=0)
        total_night_calls=st.slider('Insira o numero de ligações por noite :', min_value=0, max_value=250, value=0)
        total_night_charge=st.slider('Insira o numero de recargas por noite :', min_value=0, max_value=250, value=0)
        total_intl_minutes=st.slider('Insira o numero de minutos internacionais :', min_value=0, max_value=250, value=0)
        total_intl_calls=st.slider('Insira o numero de ligações internacionais :', min_value=0, max_value=250, value=0)
        total_intl_charge=st.slider('Insira o numero de recargas internacionais :', min_value=0, max_value=250, value=0)
        number_customer_service_calls=st.slider('Insira o numero de ligações para atendimento ao cliente :', min_value=0, max_value=250, value=0)

        # Dicionário para gerar o dataset
        input_dict={'state':state,'account_length': account_length,'area_code':area_code,'international_plan':international_plan,'voice_mail_plan':voice_mail_plan\
		,'number_vmail_messages':number_vmail_messages,'total_day_minutes':total_day_minutes,'total_day_calls':total_day_calls\
        ,'total_day_charge':total_day_charge, 'total_eve_minutes':total_eve_minutes,'total_eve_calls':total_eve_calls\
        ,'total_eve_charge':total_eve_charge,'total_night_minutes':total_night_minutes,'total_night_calls':total_night_calls\
        ,'total_night_charge':total_night_charge,'total_intl_minutes':total_intl_minutes,'total_intl_calls':total_intl_calls\
		,'total_intl_charge':total_intl_charge ,'number_customer_service_calls':number_customer_service_calls}

        # gera o dataset para o modelo
        df_inputed = pd.DataFrame([input_dict])

        if st.button('Predict'):
            st.write(df_inputed.shape)
            st.write(df_inputed)
            predicted_class, predicted_proba = pipeline_predict(df_inputed)
            st.subheader('Dados inseridos pelo usuário')
            st.write(df_inputed)
            st.write('A predição de churn deste cliente é: {} com a probabilidade {}.'.format(predicted_class, predicted_proba))
    
    if options == 'Explicabilidade':
        pipeline_predict('', 'Explicabilidade')

if __name__ == '__main__':
    main()