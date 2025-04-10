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


def cargar_dataset_digits():
    digits = load_digits()
    return digits

def escalar_datos(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

def dividir_datos(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def entrenar_modelo_logistico(X_train, y_train):
    modelo = LogisticRegression(max_iter=500)
    modelo.fit(X_train, y_train)
    return modelo

def evaluar_modelo(modelo, X_test, y_test):
    y_pred = modelo.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    print("Precisión:", accuracy)
    print("Matriz de Confusión:\n", conf_matrix)
    print("Informe de Clasificación:\n", class_report)
    return accuracy, conf_matrix, class_report

def cargar_dataset_wine():
    wine = load_wine()
    return wine

def normalizar_datos(X):
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)
    return X_normalized

def entrenar_modelo_kmeans(X, n_clusters=3):
    modelo = KMeans(n_clusters=n_clusters)
    modelo.fit(X)
    return modelo

def asignar_etiquetas(modelo, X):
    labels = modelo.labels_
    return labels

def ejecucion_supervisado():
    data = cargar_dataset_digits()
    X_scaled = escalar_datos(data.data)
    X_train, X_test, y_train, y_test = dividir_datos(X_scaled, data.target)
    modelo = entrenar_modelo_logistico(X_train, y_train)
    evaluar_modelo(modelo, X_test, y_test)

ejecucion_supervisado()

def ejecucion_no_supervisado():
    data = cargar_dataset_wine()
    X_normalized = normalizar_datos(data.data)
    modelo = entrenar_modelo_kmeans(X_normalized)
    labels = asignar_etiquetas(modelo, X_normalized)

ejecucion_no_supervisado()
