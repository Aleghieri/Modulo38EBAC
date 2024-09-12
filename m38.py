import streamlit as st
import pandas as pd
from pycaret.classification import load_model, predict_model

modelo = load_model('modelo_treinado')

def carregar_csv():
    st.header("Carregar CSV para Escoragem")
    arquivo = st.file_uploader("Selecione o arquivo CSV", type=["csv"])
    
    if arquivo is not None:
        data = pd.read_csv(arquivo)
        st.write("Dados carregados:")
        st.write(data.head())
        return data
    else:
        st.warning("Por favor, carregue um arquivo CSV.")
        return None

def preprocessamento(data):
    data.fillna(0, inplace=True)
    return data

def main():
    st.title("Escoragem de Dados com Modelo Treinado")
    
    dados = carregar_csv()

    if dados is not None:
        dados_preprocessados = preprocessamento(dados)
        
        previsoes = predict_model(modelo, data=dados_preprocessados)

        st.write("Previsões:")
        st.write(previsoes[['prediction_label', 'prediction_score']])

if __name__ == "__main__":
    main()