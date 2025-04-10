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
    pass

# 1.1. Preprocesamiento de Datos
# Escalar los datos
def escalar_datos(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled
    pass

# Dividir los datos en conjuntos de entrenamiento y prueba
def dividir_datos(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)
    pass

# 1.2. Entrenamiento del Modelo
# Entrenar un modelo de Regresión Logística con 500 iteraciones máximo.
def entrenar_modelo_logistico(X_train, y_train):
    modelo=LogisticRegression(max_iter=500)
    modelo.fit(X_train,y_train)
    return modelo
    pass

# 1.3. Evaluación del Modelo
# Realizar predicciones y evaluar el modelo
def evaluar_modelo(modelo, X_test, y_test):
    y_pred=modelo.predict(X_test)
    accuracy=accuracy_score(y_test,y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    return accuracy, conf_matrix, class_report
    pass

# Ejecución del examen
# Parte 1: Aprendizaje Supervisado
def ejecucion_supervisado():
    #cargar datos
    data = cargar_dataset_digits()
    #dividir datos
    X_train, X_test, y_train, y_test = dividir_datos(data.data, data.target)
    #escalar datos
    X_scaled = escalar_datos(X_train)
    #entrenar modelo
    modelo = entrenar_modelo_logistico(X_scaled, y_train)
    #evaluar modelo
    accuracy, conf_matrix, class_report = evaluar_modelo(modelo, X_test, y_test)
    return accuracy, conf_matrix, class_report

# Parte 2: Aprendizaje No Supervisado
# Dataset: Cargar el dataset de wine
def cargar_dataset_wine():
    return load_wine()
    pass

# 2.1. Preprocesamiento de Datos
# Normalizar los datos
def normalizar_datos(X):
    scaler=StandardScaler()
    return scaler.fit_transform(X)
    pass

# 2.2. Entrenamiento del Modelo
# Entrenar un modelo de K-Means
def entrenar_modelo_kmeans(X, n_clusters=3):
    modelo = KMeans(n_clusters=n_clusters, random_state=42)
    modelo.fit(X)
    return modelo    
    pass

# 2.3. Evaluación del Modelo
# Asignar etiquetas a los datos
def asignar_etiquetas(modelo, X):
    return modelo.predict(X)
    pass

# Visualizar los clusters
def visualizar_clusters(X, labels):
    plt.scatter(X[:, 0], X[:, 1], c=labels)
    plt.title('Visualización de Clusters')
    plt.show()
    pass

# Parte 2: Aprendizaje No Supervisado
def ejecucion_no_supervisado():
    #cargar datosd
    data = cargar_dataset_wine()
    #normalizar datos
    X_scaled = normalizar_datos(data.data)
    #entrenar modelo
    modelo = entrenar_modelo_kmeans(X_scaled)
    #asignar etiquetas
    labels = asignar_etiquetas(modelo, X_scaled)
    #ver los resultados
    visualizar_clusters(X_scaled, labels)
    return labels
    pass

if __name__ == '__main__':
    ejecucion_supervisado()
    ejecucion_no_supervisado()

