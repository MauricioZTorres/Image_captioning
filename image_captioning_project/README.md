
# Image Captioning Project

Este proyecto implementa un modelo de generación de descripciones para imágenes usando PyTorch.

## Requisitos
- Python 3.11+
- PyTorch
- CUDA (para GPU)

## Instalación
1. Clonar el repositorio:
```bash
git clone [URL_DEL_REPO]
```

2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

3. Descargar el dataset:
- Crear una carpeta `archive`
- Descargar el dataset de Flickr8k y colocarlo en la carpeta `archive`
- La estructura debe ser:
  ```
  archive/
  ├── Images/
  └── captions.txt
  ```

## Uso
1. Abrir el notebook `clasic.ipynb`
2. Ejecutar las celdas en orden
3. El modelo se guardará cada 5 épocas

## Estructura del Proyecto
- `clasic.ipynb`: Notebook principal con el código
- `dataset.py`: Clase del dataset
- `requirements.txt`: Dependencias del proyecto
