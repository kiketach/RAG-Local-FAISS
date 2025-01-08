import argparse
import json
import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv
import os
import pandas as pd

# Cargar variables de entorno
load_dotenv()

# Configurar la API key de OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

def generate_embedding(text):
    """
    Genera un embedding para un texto dado usando la API de OpenAI.
    Esto en caso que no se tenga el embedding creado previamente.
    """
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return np.array(response.data[0].embedding)

def load_data(file_path):
    """
    Carga los datos desde un archivo JSON.
    Seria el archivo original de la información.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def build_texts(data):
    """
    Construye textos a partir de los datos cargados.
    Esto con el fin de organizar el archivo cargado.
    Obviamente variará dependiendo de la estructura del archivo JSON.
    """
    texts = []
    for entry in data:
        text = f"Nombre: {entry['Nombre']}\nDescripción: {entry['Ejec_Descripcion']}\nFecha de compromiso: {entry['Ejec_Compromiso']}"
        texts.append(text)
    return texts

def load_embeddings(file_path):
    """
    Carga los embeddings desde un archivo CSV.
    En este caso el embedding fue guardado como CSV.
    """
    embeddings_df = pd.read_csv(file_path)
    return embeddings_df.to_numpy()

def create_faiss_index(embeddings):
    """
    Crea un índice FAISS a partir de los embeddings.
    """
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def mejorar_respuesta_con_gpt(hallazgos, consulta):
    """
    Mejora la respuesta usando GPT-4o-mini.
    """
    # Crear un prompt para GPT
    prompt = f"El usuario hizo la siguiente consulta: '{consulta}'\n\n"
    prompt += "A continuación se muestran los hallazgos más relevantes encontrados:\n"
    for i, hallazgo in enumerate(hallazgos, start=1):
        prompt += f"{i}. {hallazgo}\n"
    prompt += "\nPor favor, genera una respuesta clara y concisa basada en estos hallazgos."

    # Llamar a la API de OpenAI
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Eres un asistente útil que mejora respuestas basadas en hallazgos."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500  # Ajusta según sea necesario
    )

    # Devolver la respuesta generada
    return response.choices[0].message.content

def search_similar(index, texts, query, k=3):
    """
    Realiza una búsqueda de similitudes en el índice FAISS y mejora la respuesta con GPT.
    """
    query_embedding = generate_embedding(query).astype('float32')
    distances, indices = index.search(np.array([query_embedding]), k)
    
    # Obtener los hallazgos más similares
    hallazgos_encontrados = [texts[i] for i in indices[0]]
    
    # Mejorar la respuesta con GPT
    respuesta_mejorada = mejorar_respuesta_con_gpt(hallazgos_encontrados, query)
    
    # Mostrar la respuesta mejorada
    print("Respuesta mejorada:")
    print(respuesta_mejorada)

def main():
    """
    Función principal que maneja la ejecución del script.
    """
    parser = argparse.ArgumentParser(description="Sistema de búsqueda de similitudes con FAISS y OpenAI.")
    parser.add_argument('--load-data', action='store_true', help="Cargar datos desde el archivo JSON.")
    parser.add_argument('--load-embeddings', action='store_true', help="Cargar embeddings desde el archivo CSV.")
    parser.add_argument('--create-index', action='store_true', help="Crear el índice FAISS.")
    parser.add_argument('--search', type=str, help="Realizar una búsqueda con la consulta proporcionada.")
    args = parser.parse_args()

    # Aqui es donde se indica la ruta del archivo JSON
    if args.load_data:
        print("Cargando datos...")
        data = load_data('data/Hallazgos.json')
        texts = build_texts(data)
        print("Datos cargados y textos construidos.")

    # Aqui es donde se indica la ruta del embedding 
    if args.load_embeddings:
        print("Cargando embeddings...")
        embeddings = load_embeddings('data/hallazgos.csv')
        print("Embeddings cargados desde 'data/hallazgos.csv'.")

    # Crear índice FAISS
    if args.create_index:
        print("Creando índice FAISS...")
        embeddings = load_embeddings('data/hallazgos.csv')  # Cargar embeddings desde CSV
        index = create_faiss_index(embeddings)
        faiss.write_index(index, 'data/index_file.index')  # Guardar el índice
        print("Índice FAISS creado y guardado en 'data/index_file.index'.")

    # Realizar búsqueda
    if args.search:
        print(f"Realizando búsqueda para: '{args.search}'")
        index = faiss.read_index('data/index_file.index')  # Carga el índice
        data = load_data('data/Hallazgos.json')  # Carga datos originales
        texts = build_texts(data)  # Construye los textos del archivo original
        search_similar(index, texts, args.search)

if __name__ == "__main__":
    main()