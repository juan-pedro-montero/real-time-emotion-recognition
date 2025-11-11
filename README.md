# Proyecto de Red Neuronal Convolucional (CNN) - Detección de Emociones

## Resumen

Este proyecto desarrolla un sistema de reconocimiento de emociones faciales en tiempo real, basado en redes neuronales convolucionales y orientado a aplicaciones en telepsicología. El modelo se entrena utilizando el conjunto de datos AffectNet, uno de los más amplios y diversos en el ámbito del reconocimiento afectivo, y se implementa sobre la arquitectura MobileNetV2 mediante técnicas de transferencia de aprendizaje.

El flujo completo abarca desde la preparación y análisis exploratorio del dataset hasta el despliegue del modelo en un entorno de producción con captura en vivo desde webcam. Para validar su desempeño, se evaluaron múltiples configuraciones de entrenamiento, destacándose aquella que combina preprocesamiento por detección facial y fine-tuning profundo, alcanzando un rendimiento destacado con un F1-score ponderado de 0.65 y una precisión del 65% en el conjunto de validación.

El sistema es ligero, eficiente y funcional sin necesidad de GPU, lo que permite su integración en dispositivos convencionales. Como aplicación principal, se propone su uso como herramienta de apoyo para psicólogos y profesionales de la salud mental en sesiones remotas, facilitando el monitoreo emocional del paciente de forma automática y continua.

Este proyecto se divide en dos partes principales:

1. **EDA**: un notebook que analisa las imagenes de affectnet.(`Exploratory_data_analysis.ipynb`)
2. **Entrenamiento**: Implementado en cuatro notebooks de Jupyter (`Notebook_affectnetv1.ipynb` a `Notebook_affectnetv4.ipynb`), donde se entrena una CNN utilizando el conjunto de datos AffectNet.
3. **Aplicación**: Implementado en `affectnet_webcam_detector.py`, que conecta la webcam y utiliza uno de los modelos entrenados para predecir emociones en tiempo real.

## Conjunto de Datos

El conjunto de datos utilizado es [AffectNet](https://huggingface.co/datasets/chitradrishti/AffectNet/blob/main/affectnet.zip).  
Debes descargarlo y establecer la ruta correspondiente al inicio de cada notebook, donde se especifica el path para el acceso a los datos.

## Uso

### 1. Configuración del Entorno

Se recomienda utilizar un entorno virtual de Python para evitar conflictos de dependencias.

```bash
# Crear entorno virtual
python -m venv venv

# Activar el entorno virtual
# En Windows:
venv\Scripts\activate
# En Linux o macOS:
source venv/bin/activate
```

Instala las dependencias necesarias ejecutando:

```bash
pip install -r requirements.txt
```
Asegúrate de utilizar Python **3.10.4**.

### 2. EDA

Abre y ejecuta el notebook (`Exploratory_data_analysis.ipynb`).  
Antes de comenzar, ajusta la ruta al directorio donde has descargado el conjunto de datos.


### 3. Entrenamiento

Abre y ejecuta los notebooks en orden (`Notebook_affectnetv1.ipynb` hasta `Notebook_affectnetv4.ipynb`).  
Antes de comenzar, ajusta la ruta al directorio donde has descargado el conjunto de datos.


### 4. Aplicación

Para utilizar el modelo entrenado con la webcam:

```bash
python affectnet_webcam_detector.py
```

En `affectnet_webcam_detector.py`, modifica la ruta del modelo si deseas utilizar uno distinto. Por defecto, está configurado para usar:

```
best_model_finetuned_affectnet_v4.pt
```

## Notas Adicionales

- Es fundamental que la ruta al modelo entrenado esté correctamente configurada antes de ejecutar `affectnet_webcam_detector.py`.
- Asegúrate de que tu webcam esté correctamente conectada y accesible.

## Resultados obtenidos en Validación

Durante la experimentación se evaluaron seis modelos, correspondientes a cuatro configuraciones diferentes de entrenamiento. A continuación se detallan los resultados principales en términos de precisión, F1-score ponderado y AUC promedio, tomando como referencia el mejor modelo obtenido por cada configuración.

### Tabla comparativa de resultados

| Configuración | Modelo                 | Precisión | F1-score Weighted | AUC Promedio | Observaciones                                                   |
|---------------|------------------------|-----------|-------------------|--------------|-----------------------------------------------------------------|
| Config. 1     | Frozen                 | 47%       | 0.46              | 0.86         | Buen punto de partida, sin signos de overfitting                |
| Config. 1     | Fine-Tuned             | 60%       | 0.61              | 0.92         | Fine-tuning mejora rendimiento sin sobreajuste                  |
| Config. 2     | Frozen                 | 47%       | 0.46              | 0.86         | Similar a Config. 1, pero con adaptaciones a webcam             |
| Config. 2     | Fine-Tuned (época 4)   | 62%       | 0.61              | 0.92         | Buen rendimiento general, especialmente en clases clave         |
| Config. 3     | Fine-Tuned (época 4)   | 62%       | 0.63              | 0.93         | Mejor F1-score, pero signos de sobreajuste en épocas posteriores |
| Config. 4     | Fine-Tuned (época 3)   | **65%**   | **0.65**          | **0.93**     | Mejor rendimiento global, con recorte facial y entrenamiento profundo |

---

### Análisis de resultados

- **Configuraciones 1 y 2** muestran mejoras claras al aplicar fine-tuning, especialmente cuando se adaptan al entorno de uso (webcam).
- **Configuración 3** alcanza un mayor F1-score, pero a costa de una menor estabilidad, evidenciando posibles signos de overfitting.
- **Configuración 4** presenta el mejor balance entre precisión, robustez y generalización, gracias al preprocesamiento facial y el fine-tuning más profundo. Esto la convierte en la opción más adecuada para entornos reales o clínicos.


## Métricas por clase (Configuración 4  -  Modelo 6)

A continuación se detallan las métricas por clase para el modelo final entrenado con la Configuración 4. Este modelo utiliza MobileNetV2 con fine-tuning profundo y preprocesamiento basado en detección y recorte facial.

| Emoción    | Precisión | Recall | F1-score | Soporte |
|------------|-----------|--------|----------|---------|
| Anger      | 0.48      | 0.53   | 0.51     | 668     |
| Contempt   | 0.67      | 0.59   | 0.63     | 627     |
| Disgust    | 0.55      | 0.41   | 0.47     | 530     |
| Fear       | 0.51      | 0.53   | 0.52     | 654     |
| Happy      | 0.93      | 0.87   | 0.90     | 1069    |
| Neutral    | 0.81      | 0.86   | 0.83     | 1011    |
| Sad        | 0.47      | 0.57   | 0.52     | 654     |
| Surprise   | 0.58      | 0.56   | 0.57     | 834     |

**Métricas globales**:

- **Accuracy**: 65%
- **F1-score macro promedio**: 0.62
- **F1-score weighted promedio**: 0.65

### Análisis

- El modelo muestra un desempeño sólido en clases bien representadas como **happy** (f1 = 0.90) y **neutral** (f1 = 0.83), lo cual es esperable dada su mayor frecuencia y claridad visual.
- Las clases con menor rendimiento fueron **disgust** (f1 = 0.47) y **anger** (f1 = 0.51), posiblemente debido a su menor representación y ambigüedad expresiva.
- El desempeño balanceado entre precisión y recall indica una capacidad razonable de generalización sin sesgo excesivo hacia clases dominantes.

