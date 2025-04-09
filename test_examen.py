import unittest
import numpy as np
from sklearn.datasets import load_digits, load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from examen_del_segundo_parcial import (cargar_dataset_digits, escalar_datos, dividir_datos,
                                        entrenar_modelo_logistico, evaluar_modelo,
                                        cargar_dataset_wine, normalizar_datos,
                                        entrenar_modelo_kmeans, asignar_etiquetas)

# Pruebas unitarias
class TestFuncionesExamen(unittest.TestCase):

    def test_cargar_dataset_digits(self):
        data = cargar_dataset_digits()
        self.assertEqual(data.data.shape[1], 64)  # Digits dataset has 64 features

    def test_cargar_dataset_wine(self):
        data = cargar_dataset_wine()
        self.assertEqual(data.data.shape[1], 13)  # Wine dataset has 13 features

    def test_escalar_datos(self):
        data = cargar_dataset_digits()
        X_scaled = escalar_datos(data.data)
        self.assertAlmostEqual(X_scaled.mean(), 0, places=1)
        self.assertAlmostEqual(X_scaled.std(), 1, places=1)

    def test_dividir_datos(self):
        data = cargar_dataset_digits()
        X_train, X_test, y_train, y_test = dividir_datos(data.data, data.target)
        self.assertEqual(len(X_train), len(y_train))
        self.assertEqual(len(X_test), len(y_test))
        self.assertEqual(len(X_train) + len(X_test), len(data.data))

    def test_entrenar_modelo_logistico(self):
        data = cargar_dataset_digits()
        X_scaled = escalar_datos(data.data)
        X_train, X_test, y_train, y_test = dividir_datos(X_scaled, data.target)
        modelo = entrenar_modelo_logistico(X_train, y_train)
        self.assertIsInstance(modelo, LogisticRegression)

    def test_evaluar_modelo(self):
        data = cargar_dataset_digits()
        X_scaled = escalar_datos(data.data)
        X_train, X_test, y_train, y_test = dividir_datos(X_scaled, data.target)
        modelo = entrenar_modelo_logistico(X_train, y_train)
        accuracy, conf_matrix, class_report = evaluar_modelo(modelo, X_test, y_test)
        self.assertGreaterEqual(accuracy, 0)
        self.assertIsInstance(conf_matrix, np.ndarray)
        self.assertIsInstance(class_report, str)

    def test_normalizar_datos(self):
        data = cargar_dataset_wine()
        X_scaled = normalizar_datos(data.data)
        self.assertAlmostEqual(X_scaled.mean(), 0, places=1)
        self.assertAlmostEqual(X_scaled.std(), 1, places=1)

    def test_entrenar_modelo_kmeans(self):
        data = cargar_dataset_wine()
        X_scaled = normalizar_datos(data.data)
        modelo = entrenar_modelo_kmeans(X_scaled)
        self.assertIsInstance(modelo, KMeans)

    def test_asignar_etiquetas(self):
        data = cargar_dataset_wine()
        X_scaled = normalizar_datos(data.data)
        modelo = entrenar_modelo_kmeans(X_scaled)
        labels = asignar_etiquetas(modelo, X_scaled)
        self.assertEqual(len(labels), len(data.data))

    def test_funcional_aprendizaje_supervisado(self):
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
        self.assertGreaterEqual(accuracy, 0.8)  # Verificar que la precisión sea razonablemente alta
        self.assertIsInstance(conf_matrix, np.ndarray)
        self.assertIsInstance(class_report, str)
        print(f'Accuracy: {accuracy}')
        print('Confusion Matrix:')
        print(conf_matrix)
        print('Classification Report:')
        print(class_report)        

if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)