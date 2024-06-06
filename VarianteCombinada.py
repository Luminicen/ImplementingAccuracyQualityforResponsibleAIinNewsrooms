import PyPDF2
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import gensim
import pandas as pd
from gensim import corpora
def clean(doc,stop,exclude,lemma):
    stop_free = " ".join([i for i in doc.lower().split()if i not in stop])
    punc_free = "".join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized
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
def format_topics_sentences(ldamodel,corpus):
    sent_topics_df = pd.DataFrame()
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = " ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df._append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']
    return sent_topics_df
def identificarTopicos(doc_complete) :
    stop = set(stopwords.words('spanish'))
    exclude = set(string.punctuation)
    lemma = WordNetLemmatizer()
    doc_clean = [clean(doc,stop,exclude,lemma).split() for doc in doc_complete]
    dictionary = corpora.Dictionary(doc_clean)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
    Lda = gensim.models.ldamodel.LdaModel
    ldamodel = Lda(doc_term_matrix, num_topics=3, id2word =
    dictionary, passes=50)
    topicos = ldamodel.show_topics(formatted=False)
    result = format_topics_sentences(ldamodel,doc_term_matrix)
    return result   
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
entidades_p = identificarTopicos(entidades_pivote)["Topic_Keywords"][0]
comparacion=[]
for i in range(1,16):
    indice = numero(i)
    noticia = leer_pdf("./noticias/"+indice+".txt")
    entidades_noticia = extraer_entidades(nlp(noticia))
    entidades_n = identificarTopicos(entidades_noticia)["Topic_Keywords"][0]
    similitud = process_jaccardMethod(entidades_p,entidades_n)
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

