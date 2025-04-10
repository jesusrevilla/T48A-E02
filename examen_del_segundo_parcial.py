# examen_del_segundo_parcial.py

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

# Cargar el dataset de digits
def cargar_dataset_digits():
    return load_digits()

# Escalar los datos
def escalar_datos(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

# Dividir los datos en conjuntos de entrenamiento y prueba
def dividir_datos(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Entrenar un modelo de Regresión Logística
def entrenar_modelo_logistico(X_train, y_train):
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)
    return model

# Evaluar el modelo y devolver métricas
def evaluar_modelo(modelo, X_test, y_test):
    y_pred = modelo.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    return accuracy, conf_matrix, class_report

# Parte 2: Aprendizaje No Supervisado

# Cargar el dataset de wine
def cargar_dataset_wine():
    return load_wine()

# Normalizar los datos
def normalizar_datos(X):
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)
    return X_norm

# Entrenar un modelo de K-Means
def entrenar_modelo_kmeans(X, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)
    return kmeans

# Asignar etiquetas a los datos
def asignar_etiquetas(modelo, X):
    labels = modelo.predict(X)
    return labels

# Visualizar los clusters
def visualizar_clusters(X, labels):
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Clusters')
    plt.show()

# Función de ejecución del aprendizaje supervisado
def ejecucion_supervisado():
    data = cargar_dataset_digits()
    X_scaled = escalar_datos(data.data)
    X_train, X_test, y_train, y_test = dividir_datos(X_scaled, data.target)
    modelo = entrenar_modelo_logistico(X_train, y_train)
    accuracy, conf_matrix, class_report = evaluar_modelo(modelo, X_test, y_test)
    print(f"Accuracy: {accuracy}")
    print("Confusion Matrix:\n", conf_matrix)
    print("Classification Report:\n", class_report)

# Función de ejecución del aprendizaje no supervisado
def ejecucion_no_supervisado():
    data = cargar_dataset_wine()
    X_norm = normalizar_datos(data.data)
    modelo = entrenar_modelo_kmeans(X_norm)
    labels = asignar_etiquetas(modelo, X_norm)
    visualizar_clusters(X_norm, labels)


