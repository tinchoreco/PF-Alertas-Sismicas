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
@app.get("/clasificar_sismo/{magnitud}/{gap}/{intensidad}/{hipocentro}")
def clasificar_sismo(magnitud: float, gap: float, intensidad: float, hipocentro: float):
    # Realizar la clasificación utilizando el modelo entrenado
    clasificacion = model.predict([[magnitud, gap, intensidad, hipocentro]])

    # Devolver la clasificación como respuesta
    return {"clasificacion": clasificacion[0]}

# Iniciar el servidor
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
