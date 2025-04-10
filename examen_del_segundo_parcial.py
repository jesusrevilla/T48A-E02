# Examen Práctico de Minería de Datos
# Universidad Politécnica de San Luis Potosí
# Curso: Minería de Datos
# Tema: Aprendizaje Supervisado y No Supervisado

# Importar las librerías necesarias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import load_digits, load_wine
from sklearn.preprocessing import StandardScaler

# Parte 1: Aprendizaje Supervisado
# Dataset: Cargar el dataset de digits
def cargar_dataset_digits():
  return load_digits()


# 1.1. Preprocesamiento de Datos
# Escalar los datos
def escalar_datos(X):#return x_scaled
  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X)
  
  return X_scaled
  

# Dividir los datos en conjuntos de entrenamiento y prueba
def dividir_datos(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)

# 1.2. Entrenamiento del Modelo
# Entrenar un modelo de Regresión Logística con 500 iteraciones máximo.
def entrenar_modelo_logistico(X_train, y_train):
  logreg = LogisticRegression(max_iter=500)
  logreg.fit(X_train, y_train)
  return logreg
    

# 1.3. Evaluación del Modelo
# Realizar predicciones y evaluar el modelo
def evaluar_modelo(modelo, X_test, y_test):
  y_pred = modelo.predict(X_test)
  accuracy = accuracy_score(y_test, y_pred)
  conf_matrix = confusion_matrix(y_test, y_pred)
  classification_rep = classification_report(y_test, y_pred)
  return accuracy, conf_matrix, classification_rep
 
    

# Ejecución del examen
# Parte 1: Aprendizaje Supervisado
def ejecucion_supervisado():
  #Cargar datos
  digits = load_digits()
  X = digits.data
  y = digits.target

  #Escalar datos
  X_scaled = escalar_datos(X)
 
  #Dividir datos
  X_train, X_test, y_train, y_test = dividir_datos(X_scaled, y)
 
  #Entrenar modelo
  logreg = entrenar_modelo_logistico(X_train, y_train)

  #Evaluar modelo
  accuracy, conf_matrix, classification_rep = evaluar_modelo(logreg, X_test, y_test)
  print("Accuracy:", accuracy)
  print("Confusion Matrix:\n", conf_matrix)
  print("Classification Report:\n", classification_rep)
  

  
    

ejecucion_supervisado()

# Parte 2: Aprendizaje No Supervisado
# Dataset: Cargar el dataset de wine
def cargar_dataset_wine():
    return load_wine()
    

# 2.1. Preprocesamiento de Datos
# Normalizar los datos
def normalizar_datos(X):
  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X)
  return X_scaled
    

# 2.2. Entrenamiento del Modelo
# Entrenar un modelo de K-Means
def entrenar_modelo_kmeans(X, n_clusters=3):
  kmeans = KMeans(n_clusters=n_clusters, random_state=42)
  kmeans.fit(X)
  return kmeans
    

# 2.3. Evaluación del Modelo
# Asignar etiquetas a los datos
def asignar_etiquetas(modelo, X):
  labels = modelo.predict(X)
  return labels
    

# Visualizar los clusters
def visualizar_clusters(X, labels):
  return plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
    

# Parte 2: Aprendizaje No Supervisado
def ejecucion_no_supervisado():

  #Cargar los datos
  wine = load_wine()
  X = wine.data
  y = wine.target

  #Normalizar los datos
  X_scaled = normalizar_datos(X)

  #Entrenar el modelo
  kmeans = entrenar_modelo_kmeans(X_scaled)


  #Asignar eriqueta a los datos
  labels = asignar_etiquetas(kmeans, X_scaled)

  #Visualizar los clusters
  visualizar_clusters(X_scaled, labels)
  plt.show()
 

    
ejecucion_no_supervisado()
