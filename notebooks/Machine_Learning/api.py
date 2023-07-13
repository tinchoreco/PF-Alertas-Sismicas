
import pandas as pd
import pickle

# Carga el modelo desde el archivo

filename = "D://Git Hub Clonaciones//PF-Alertas-Sismicas//notebooks//Machine_Learning//kmeans_model.pkl"

#filename = "..//notebooks//Machine_Learning//api.py"
with open(filename, 'rb') as file:
    kmeans_loaded = pickle.load(file)

# Crea un DataFrame con los nuevos datos
nuevos_datos = pd.DataFrame({'Intensidad': [4.3], 'Magnitud': [4.5]})

# Aplica el modelo a los nuevos datos
labels_pred = kmeans_loaded.predict(nuevos_datos)
nuevos_datos["labels_pred"] = labels_pred
# Imprime las etiquetas predichas
print(labels_pred)
print(nuevos_datos)



