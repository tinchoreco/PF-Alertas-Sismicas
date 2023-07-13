from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Definición de la estructura de datos para recibir los parámetros
class Sismo(BaseModel):
    Magnitud: float
    GAP: float
    Intensidad: float
    Hipocentro: float

# Carga del modelo entrenado
model = joblib.load("kmeans_model.pkl")

# Creación de la aplicación FastAPI
app = FastAPI()

# Definición del endpoint para la clasificación
@app.get("/clasificar_sismo")
def clasificar_sismo(sismo: Sismo):
    # Obtención de los parámetros del sismo
    magnitud = sismo.Magnitud
    gap = sismo.GAP
    intensidad = sismo.Intensidad
    hipocentro = sismo.Hipocentro
    
    # Realizar la clasificación utilizando el modelo entrenado
    clasificacion = model.predict([[magnitud, gap, intensidad, hipocentro]])
    
    # Devolver la clasificación como respuesta
    return {"clasificacion": clasificacion[0]}

# Iniciar el servidor
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
