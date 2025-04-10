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

# 1.1. Preprocesamiento de Datos
# Escalar los datos
def escalar_datos(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

# Dividir los datos en conjuntos de entrenamiento y prueba
def dividir_datos(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    return X_train, X_test, y_train, y_test

# 1.2. Entrenamiento del Modelo
# Entrenar un modelo de Regresión Logística con 500 iteraciones máximo.
def entrenar_modelo_logistico(X_train, y_train):
    modelo_logistico = LogisticRegression(max_iter=500)
    modelo_logistico.fit(X_train, y_train)
    return modelo_logistico

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
    # Cargar y escalar el dataset
    data = cargar_dataset_digits()
    X_scaled = escalar_datos(data.data)

    # Dividir los datos
    X_train, X_test, y_train, y_test = dividir_datos(X_scaled, data.target)

    # Entrenar el modelo
    modelo = entrenar_modelo_logistico(X_train, y_train)

    # Evaluar el modelo
    accuracy, conf_matrix, class_report = evaluar_modelo(modelo, X_test, y_test)

    # Verificaciones
    print("Accuracy:", accuracy)
    print(f'Accuracy: {accuracy}')
    print('Confusion Matrix:')
    print(conf_matrix)
    print('Classification Report:')
    print(class_report)

ejecucion_supervisado()

# Parte 2: Aprendizaje No Supervisado
# Dataset: Cargar el dataset de wine
def cargar_dataset_wine():
    data = load_wine()
    return data

# 2.1. Preprocesamiento de Datos
# Normalizar los datos
def normalizar_datos(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

# 2.2. Entrenamiento del Modelo
# Entrenar un modelo de K-Means
def entrenar_modelo_kmeans(X, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(X)
    return kmeans

# 2.3. Evaluación del Modelo
# Asignar etiquetas a los datos
def asignar_etiquetas(modelo, X):
    labels = modelo.predict(X)
    return labels

# Visualizar los clusters
def visualizar_clusters(X, labels):
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Clusters')
    plt.show()

# Parte 2: Aprendizaje No Supervisado
def ejecucion_no_supervisado():
    # Cargar y normalizar el dataset
    data = cargar_dataset_wine()
    X_scaled = normalizar_datos(data.data)
    
    # Entrenar el modelo
    modelo = entrenar_modelo_kmeans(X_scaled)
    
    # Asignar etiquetas a los datos
    labels = asignar_etiquetas(modelo, X_scaled)
    
    # Visualizar los clusters
    visualizar_clusters(X_scaled, labels)

ejecucion_no_supervisado()
