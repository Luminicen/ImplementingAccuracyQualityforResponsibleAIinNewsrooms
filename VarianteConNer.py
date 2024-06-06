import PyPDF2
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
def leer_pdf(nombre_archivo):
    texto_completo = "" 
    with open(nombre_archivo, 'r',encoding='utf-8') as archivo:
        contenido = archivo.read()
        texto_completo =contenido
    return texto_completo
def extraer_entidades(doc):
    entidades_filtradas = [entidad for entidad in doc.ents if entidad.label_ in ['PERSON', 'ORG','LOC','GPE']]
    lista = []
    for i in entidades_filtradas:
        lista.append(i.text.lower())
    return lista
def numero(indice):
    aux=''
    while(len(aux+str(indice))!= 3):
        aux = aux + '0'
    
    return (aux+str(indice))
def sacar_stopwords(lista_palabras):
    stopwords_es = set(stopwords.words('spanish'))
    palabras_filtradas = [palabra for palabra in lista_palabras if palabra.lower() not in stopwords_es]
    return palabras_filtradas
def process_jaccardMethod(i,j):
    """
    Jaccard Similarity Method
    """
    if i == None or i == None:
        return 0
    set_i = set(i)
    set_j = set(j)
    intersection = set_i.intersection(set_j)
    union = set_i.union(set_j)
    return (len(intersection) / len(union))
nltk.download('stopwords')
ruta_pivote = "./noticias/pivote.txt"
pivote = leer_pdf(ruta_pivote)
nlp = spacy.load("es_core_news_sm")
doc = nlp(pivote)
entidades_pivote = extraer_entidades(doc)
comparacion=[]
for i in range(1,16):
    indice = numero(i)
    noticia = leer_pdf("./noticias/"+indice+".txt")
    entidades_noticia = extraer_entidades(nlp(noticia))
    similitud = process_jaccardMethod(entidades_pivote,entidades_noticia)
    comparacion.append((indice,similitud))
resultados_ordenados = sorted(comparacion, key=lambda x: x[1], reverse=True)
df = pd.read_excel("./noticias/noticias.xlsx", sheet_name='Hoja1')
final = []
from prettytable import PrettyTable
tabla = PrettyTable()
tabla.field_names = ["Titulo Noticia", "Indice de similitud"]
for z in range(15):
    num = int(resultados_ordenados[z][0].lstrip('0')) - 1
    tabla.add_row([str(df.iloc[num,1]),resultados_ordenados[z][1]])
print(tabla)



