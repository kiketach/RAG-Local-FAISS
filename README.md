# RAG Local con FAISS

Este proyecto (es mi primer proyecto serio con RAG de manera local - al menos un poco-) implementa un sistema de búsqueda de similitudes utilizando FAISS (Facebook AI Similarity Search) y embeddings generados por OpenAI. El sistema permite buscar hallazgos similares en una base de datos local.
Las respuestas son apoyadas con gpt-4o-mini para darle resultado optimo al usuario

## Características principales
- **Búsqueda eficiente**: Utiliza FAISS para encontrar hallazgos similares en grandes conjuntos de datos.
- **Integración con OpenAI**: Genera embeddings utilizando el modelo `text-embedding-ada-002`.
- **Flexibilidad**: Puede trabajar con datos estructurados (SQL Server) o archivos JSON.

## Requisitos
- Python 3.8 o superior.
- Librerías: `openai`, `faiss-cpu`, `numpy`, `pandas`, `pyodbc` (para conexión a SQL Server).

## Instalación
1. Clona el repositorio:
   git clone https://github.com/kiketach/RAG-Local-FAISS.git

2. Instala las dependencias:
   pip install -r requirements.txt

## Uso
El script principal (main.py) permite ejecutar varios pasos de manera independiente o en secuencia.

1. **Cargar datos desde Hallazgos.json**
python main.py --load-data

2. **Cargar embeddings desde hallazgos.csv**
python main.py --load-embeddings

3. **Crear el índice FAISS (index_file.index)**
python main.py --create-index

4. **Realizar una búsqueda**
python main.py --search "Conoces el hallazgo CONSOLIDACIÓN PARCIAL DE LAS INVERSIONES TEMPORALES"

**Ejecutar todo en secuencia**
python main.py --load-data --load-embeddings --create-index --search "Consulta de ejemplo"