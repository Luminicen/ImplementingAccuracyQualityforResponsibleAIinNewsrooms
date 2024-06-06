import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import gensim
import pandas as pd
from gensim import corpora
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import numpy as np
from prettytable import PrettyTable

# Descargar recursos necesarios
nltk.download('stopwords')
nltk.download('wordnet')

# Función para extraer temas usando TF-IDF y SVD
def extract_topics_tfidf(texts, num_topics=None):
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(texts)
    max_components = min(tfidf_matrix.shape) - 1
    if max_components < 1:
        raise ValueError("No se puede extraer temas de un documento vacío o con muy poca información.")
    
    if num_topics is None or num_topics > max_components:
        num_topics = max_components
    
    if num_topics < 1:
        num_topics = 1

    svd = TruncatedSVD(n_components=num_topics)
    svd_topic_vectors = svd.fit_transform(tfidf_matrix)
    terms = tfidf.get_feature_names_out()
    topics = []
    for i, topic in enumerate(svd.components_):
        top_words_idx = topic.argsort()[-10:][::-1]
        top_words = [terms[idx] for idx in top_words_idx]
        topics.append(top_words)
    return topics

# Función para leer archivos de texto
def leer_archivo(nombre_archivo):
    with open(nombre_archivo, 'r', encoding='utf-8') as archivo:
        contenido = archivo.read()
    return contenido

# Función para limpiar texto
def clean(doc):
    stop = set(stopwords.words('spanish'))
    exclude = set(string.punctuation)
    lemma = WordNetLemmatizer()
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = "".join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

# Función para formatear temas y frases
def format_topics_sentences(ldamodel, corpus):
    sent_topics_df = pd.DataFrame()
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df._append(pd.Series([int(topic_num), round(prop_topic, 4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']
    return sent_topics_df

# Función para identificar temas en documentos completos

# Función para generar número con formato específico
def numero(indice):
    return f"{indice:03}"

# Método de similitud de Jaccard
def process_jaccardMethod(i, j):
    if i is None or j is None:
        return 0
    set_i = set(i)
    set_j = set(j)
    intersection = set_i.intersection(set_j)
    union = set_i.union(set_j)
    return len(intersection) / len(union)

# Función para filtrar tokens en un texto usando spaCy
def filtrar(nlp, text):
    # Convierte la lista de palabras en una cadena de texto separada por espacios
    textito = " ".join(text)
    
    # Procesa el texto con spaCy
    doc = nlp(textito)
    
    # Filtra los tokens, eliminando signos de puntuación y adverbios
    lista = [token.text for token in doc if (not token.is_punct and token.pos_ != 'ADV')]
    
    # Convierte los tokens filtrados a minúsculas y excluye comas
    resul = [i.lower() for i in lista if i != ',']
    
    return resul

# Inicializar spaCy
nlp = spacy.load("es_core_news_sm")

# Leer archivo pivote
ruta_pivote = "./noticias/pivote.txt"
pivote = leer_archivo(ruta_pivote)
topicos_pivote = extract_topics_tfidf([clean(pivote),clean(pivote)])

# Filtrar tópicos pivote
topicos_p = filtrar(nlp, topicos_pivote[0])

# Leer y procesar noticias
texts = []
texts.append(clean(pivote))
for i in range(1, 16):
    indice = numero(i)
    noti = leer_archivo(f"./noticias/{indice}.txt")
    texts.append(clean(noti))

# Extraer tópicos de noticias
data = extract_topics_tfidf(texts)
# Comparar similitud de Jaccard
comparacion = []
for i in range(15):
    topicos_n = filtrar(nlp, data[i])
    #print(topicos_p)
    #print(topicos_n)
    #print("------")
    similitud = process_jaccardMethod(topicos_p, topicos_n)
    comparacion.append((numero(i+1), similitud))
# Ordenar resultados por similitud
resultados_ordenados = sorted(comparacion, key=lambda x: x[1], reverse=True)

# Leer archivo Excel
df = pd.read_excel("./noticias/noticias.xlsx", sheet_name='Hoja1')

# Generar tabla de resultados
tabla = PrettyTable()
tabla.field_names = ["Titulo Noticia", "Indice de similitud", "Indice de diferencia"]
for z in range(15):
    num = int(resultados_ordenados[z][0].lstrip('0')) - 1
    indice_similitud = resultados_ordenados[z][1]
    indice_diferencia = 1 - indice_similitud
    tabla.add_row([str(df.iloc[num, 1]), indice_similitud, indice_diferencia])

print(tabla)
