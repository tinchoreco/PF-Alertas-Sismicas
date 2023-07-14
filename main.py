from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Definición de la estructura de datos para recibir los parámetros
class Sismo(BaseModel):
    Magnitud: float
    Intensidad: float

# Carga del modelo entrenado
model = joblib.load("kmeans_model.pkl")

# Creacion array respuesta
respuesta = ["""Los sismos en este rango tienen una intensidad perceptible 
que va desde niveles bajos hasta niveles ampliamente perceptibles en el área afectada. 
Pueden provocar daños menores en estructuras, 
como grietas en los muros y caída de revestimientos, 
y ser percibidos por un número variable de personas, 
desde unas pocas en reposo y en posición tranquila 
hasta todas las personas en el área afectada.""",
"""Los sismos en este rango tienen una intensidad que generalmente no se percibe, 
excepto en condiciones muy favorables, 
hasta niveles que a menudo se perciben, 
pero rara vez causan daños. 
En términos de magnitud, 
van desde temblores que se sienten como vibraciones menores 
hasta sismos que pueden causar daños menores en estructuras.""",
"""los sismos en este rango tienen una intensidad que generalmente no se percibe, 
pero que en condiciones favorables puede ser percibida por unas pocas personas en reposo 
y en posición tranquila. 
Además, la magnitud de estos sismos va desde temblores de vibración menor 
hasta la capacidad de causar una gran cantidad de daños en áreas habitadas.""",
"""los sismos en este rango tienen una intensidad que a menudo se percibe 
y puede causar una gran cantidad de daños en áreas habitadas, 
junto con una magnitud que va desde temblores menores 
hasta la capacidad de causar daños significativos en estructuras.""",
"""los sismos en este rango tienen una intensidad 
que puede causar daños significativos en áreas más grandes 
y una magnitud que va desde la capacidad de causar daños significativos
 en estructuras hasta la posibilidad de ocasionar daños extensos 
 e incluso colapso total de edificios."""]
 

# Creación de la aplicación FastAPI
app = FastAPI()

# Definición del endpoint para la clasificación
@app.get("/clasificar_sismo/{magnitud}/{intensidad}")
def clasificar_sismo(magnitud: float, intensidad: float):
    # Realizar la clasificación utilizando el modelo entrenado
    clasificacion = model.predict([[magnitud, intensidad]])

    # Convertir el resultado en un tipo de datos nativo de Python
    clasificacion = np.asscalar(clasificacion)

    # Devolver la clasificación como respuesta
    return {"clasificacion": clasificacion, "texto": respuesta[clasificacion]}

# Iniciar el servidor
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
