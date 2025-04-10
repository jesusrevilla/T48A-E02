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
 
def cargar_dataset_digits():
    return load_digits()
 
def escalar_datos(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled
 
def dividir_datos(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)
 
def entrenar_modelo_logistico(X_train, y_train):
    modelo = LogisticRegression(max_iter=500)
    modelo.fit(X_train, y_train)
    return modelo
 
def evaluar_modelo(modelo, X_test, y_test):
    y_pred = modelo.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    return accuracy, conf_matrix, class_report
 
def ejecucion_supervisado():
    data = cargar_dataset_digits()
    X_scaled = escalar_datos(data.data)
    X_train, X_test, y_train, y_test = dividir_datos(X_scaled, data.target)
    modelo = entrenar_modelo_logistico(X_train, y_train)
    accuracy, conf_matrix, class_report = evaluar_modelo(modelo, X_test, y_test)
    print(f"Accuracy: {accuracy}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(class_report)
 
# Parte 2: Aprendizaje No Supervisado
 
def cargar_dataset_wine():
    return load_wine()
 
def normalizar_datos(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled
 
def entrenar_modelo_kmeans(X, n_clusters=3):
    modelo = KMeans(n_clusters=n_clusters, random_state=42)
    modelo.fit(X)
    return modelo
 
def asignar_etiquetas(modelo, X):
    return modelo.predict(X)
 
def visualizar_clusters(X, labels):
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', edgecolor='k')
    plt.title("Clusters con KMeans")
    plt.xlabel("Componente 1")
    plt.ylabel("Componente 2")
    plt.show()
 
def ejecucion_no_supervisado():
    data = cargar_dataset_wine()
    X_scaled = normalizar_datos(data.data)
    modelo = entrenar_modelo_kmeans(X_scaled)
    labels = asignar_etiquetas(modelo, X_scaled)
    # Para visualización, reducimos a 2 dimensiones
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    visualizar_clusters(X_pca, labels)
 
