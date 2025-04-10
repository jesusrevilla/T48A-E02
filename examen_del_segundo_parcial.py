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
    data = load_digits()
    return data
    pass

# 1.1. Preprocesamiento de Datos
# Escalar los datos
def escalar_datos(X):
    x = X.data
    y = X.target
    return x, y
    pass

# Dividir los datos en conjuntos de entrenamiento y prueba
def dividir_datos(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
    pass

# 1.2. Entrenamiento del Modelo
# Entrenar un modelo de Regresión Logística con 500 iteraciones máximo.
def entrenar_modelo_logistico(X_train, y_train):
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)
    return model
    pass

# 1.3. Evaluación del Modelo
# Realizar predicciones y evaluar el modelo
def evaluar_modelo(modelo, X_test, y_test):
    y_pred = modelo.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print("Precision:", accuracy)
    print("Matriz de confusion:\n", confusion)
    print("Reporte de clasificacion:\n", report)
    pass

# Ejecución del examen
# Parte 1: Aprendizaje Supervisado
def ejecucion_supervisado():
    info = cargar_dataset_digits()
    datos = escalar_datos(info)
    SDatos = dividir_datos(datos[0],datos[1])
    Modelo = entrenar_modelo_logistico(SDatos[0],SDatos[2])
    evaluar_modelo(Modelo,SDatos[1],SDatos[3])
    pass

ejecucion_supervisado()

# Parte 2: Aprendizaje No Supervisado
# Dataset: Cargar el dataset de wine
def cargar_dataset_wine():
    data2 = load_wine()
    return data2
    pass

# 2.1. Preprocesamiento de Datos
# Normalizar los datos
def normalizar_datos(X):
    x = X.data
    y = X.target
    return x, y
    pass

# 2.2. Entrenamiento del Modelo
# Entrenar un modelo de K-Means
def entrenar_modelo_kmeans(X, n_clusters=3):
    model = KMeans(n_clusters=n_clusters)
    model.fit(X)
    return model
    pass

# 2.3. Evaluación del Modelo
# Asignar etiquetas a los datos
def asignar_etiquetas(modelo, X):
    labels = modelo.predict(X)
    return labels
    pass

# Visualizar los clusters
def visualizar_clusters(X, labels):
    plot = plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
    pass

# Parte 2: Aprendizaje No Supervisado
def ejecucion_no_supervisado():
    DSet = cargar_dataset_wine()
    DNorm = normalizar_datos(DSet)
    Modelo = entrenar_modelo_kmeans(DNorm[0],3)
    Etiqueta = asignar_etiquetas(Modelo, DNorm[0])
    visualizar_clusters(DNorm[0],Etiqueta) 
    pass
ejecucion_no_supervisado()

