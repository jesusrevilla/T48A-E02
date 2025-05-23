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
    pass

# 1.1. Preprocesamiento de Datos
# Escalar los datos
def escalar_datos(X):
    pass

# Dividir los datos en conjuntos de entrenamiento y prueba
def dividir_datos(X, y):
    pass

# 1.2. Entrenamiento del Modelo
# Entrenar un modelo de Regresión Logística con 500 iteraciones máximo.
def entrenar_modelo_logistico(X_train, y_train):
    pass

# 1.3. Evaluación del Modelo
# Realizar predicciones y evaluar el modelo
def evaluar_modelo(modelo, X_test, y_test):
    pass

# Ejecución del examen
# Parte 1: Aprendizaje Supervisado
def ejecucion_supervisado():
    pass

ejecucion_supervisado()

# Parte 2: Aprendizaje No Supervisado
# Dataset: Cargar el dataset de wine
def cargar_dataset_wine():
    pass

# 2.1. Preprocesamiento de Datos
# Normalizar los datos
def normalizar_datos(X):
    pass

# 2.2. Entrenamiento del Modelo
# Entrenar un modelo de K-Means
def entrenar_modelo_kmeans(X, n_clusters=3):
    pass

# 2.3. Evaluación del Modelo
# Asignar etiquetas a los datos
def asignar_etiquetas(modelo, X):
    pass

# Visualizar los clusters
def visualizar_clusters(X, labels):
    pass

# Parte 2: Aprendizaje No Supervisado
def ejecucion_no_supervisado():
    pass
ejecucion_no_supervisado()

