
import streamlit as st
import numpy as np
import pandas as pd
from joblib import dump, load

regre = load('filename.joblib')

def main():
    st.title("APPLICATION MOBILE POUR LA DETECTION DE DIABETE")
    st.subheader("(AUTEUR: Mr.DJEGUI_WAGUE)")

    def diabete_prediction(entree_data):
        tableau_numpy = np.array(entree_data)
        input_data_reshape = tableau_numpy.reshape(1, -1)
        prediction = regre.predict(input_data_reshape)

        if (prediction[0]) == 1:
            return " La personne est  diabetique"
        else:
            return "La personne n'est pas  diabetique"


    #   on crée un fonction aui permet de telecharger nos données
    @st.cache(persist=True)
    def load_data():
        data=pd.read_csv("diabetes.csv")
        return data
#   affichage de la dataset
    df= load_data()
    df_sample=  df.sample(100)
    if st.sidebar.checkbox("AFFICHER LES DONNEES BRUTE" ,False):
            st.subheader("Jeu de données diabete : echantillon de 100 observations ")
            st.write(df_sample)


    Pregnancies= st.number_input('Nombre de fois enceinte')
    Glucose = st.number_input('Taux de glucose')
    BloodPressure = st.number_input('Pression arterielle')
    SkinThickness = st.number_input('Epaisseur de Peau')
    Insulin=st.number_input("INSULIN")
    BMI= st.number_input('Indice de masse corporelle')
    DiabetesPedigreeFunction= st.number_input('FonctionPedigreeDiabete')
    Age=st.number_input('Votre age ')


    diagnostique=""

    if st.button("resultat_du_test_diabete"):
        diagnostique =diabete_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
    st.success( diagnostique )


if __name__ == "__main__":
    main()


#   streamlit run app.py





