import numpy as np

def calcRSS(prediction_vector, real_vector):
    """prediction_vector contiene il vettore delle predizioni.
        real_vector contiene il vettore dei valori attesi"""
    return np.sum((real_vector-prediction_vector)**2)

def calcRSE(prediction_vector, real_vector, feature_matrix):
    """prediction_vector contiene il vettore delle predizioni.
    real_vector contiene il vettore dei valori attesi
    feature_matrix è una matrice dove n è il numero di sample e d è il numero di feature"""
    n, d = feature_matrix.shape
    return np.sqrt(1/(n-d-calcRSS(prediction_vector, real_vector)))

def calcMSE(prediction_vector, real_vector, feature_matrix):
    """prediction_vector contiene il vettore delle predizioni.
    real_vector contiene il vettore dei valori attesi
    feature_matrix è una matrice dove n è il numero di sample e d è il numero di feature"""
    n, d = feature_matrix.shape
    return np.divide(calcRSS(prediction_vector, real_vector),n)

def calcRMSE(prediction_vector, real_vector, feature_matrix):
    """prediction_vector contiene il vettore delle predizioni.
    real_vector contiene il vettore dei valori attesi
    feature_matrix è una matrice dove n è il numero di sample e d è il numero di feature"""
    return np.sqrt(calcMSE(prediction_vector, real_vector, feature_matrix))

def calcMAE(prediction_vector, real_vector, feature_matrix):
    """prediction_vector contiene il vettore delle predizioni.
    real_vector contiene il vettore dei valori attesi
    feature_matrix è una matrice dove n è il numero di sample e d è il numero di feature"""
    n, d = feature_matrix.shape
    return np.divide(np.sum(abs(real_vector-prediction_vector)), n)
