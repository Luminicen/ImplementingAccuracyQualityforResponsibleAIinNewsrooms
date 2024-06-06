from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import gensim
import pandas as pd
from gensim import corpora
import nltk
import spacy
def leer_pdf(nombre_archivo):
    texto_completo = "" 
    with open(nombre_archivo, 'r',encoding='utf-8') as archivo:
        contenido = archivo.read()
        texto_completo =contenido
    return texto_completo
def clean(doc,stop,exclude,lemma):
    stop_free = " ".join([i for i in doc.lower().split()if i not in stop])
    punc_free = "".join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized
def format_topics_sentences(ldamodel,corpus):
    sent_topics_df = pd.DataFrame()
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df._append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']
    return sent_topics_df
def identificarTopicos(doc_complete) :
    stop = set(stopwords.words('german'))
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
def numero(indice):
    aux=''
    while(len(aux+str(indice))!= 3):
        aux = aux + '0'
    
    return (aux+str(indice))
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
def filtrar(nlp,text):
    doc = nlp(text)
    lista =  [token.text for token in doc if (not(token.pos_ == 'ADV') or (not token.is_punct))]
    resul =[]
    for i in lista:
        if i != ',':
            resul.append(i.lower())
    return  resul
nltk.download('stopwords')
ruta_pivote = "./noticias_alem/pivote.txt"
pivote = leer_pdf(ruta_pivote)
topicos_pivote = identificarTopicos([pivote])
comparacion=[]
nlp = spacy.load("es_core_news_sm")
doc = nlp(topicos_pivote["Topic_Keywords"][0])
topicos_p= filtrar(nlp,topicos_pivote["Topic_Keywords"][0])
for i in range(1,15):
    indice = numero(i)
    noti = leer_pdf("./noticias_alem/"+indice+".txt")
    topicos_noticia = identificarTopicos([noti])
    topicos_n = filtrar(nlp,topicos_noticia["Topic_Keywords"][0])
    similitud = process_jaccardMethod(topicos_p,topicos_n)
    comparacion.append((indice,similitud))
resultados_ordenados = sorted(comparacion, key=lambda x: x[1], reverse=True)
df = pd.read_excel("./noticias_alem/noticias.xlsx", sheet_name='Hoja1')
final = []
from prettytable import PrettyTable
tabla = PrettyTable()
tabla.field_names = ["Titulo Noticia", "Indice de similitud"]
for z in range(14):
    num = int(resultados_ordenados[z][0].lstrip('0')) - 1
    tabla.add_row([str(df.iloc[num,2]),resultados_ordenados[z][1]])
print(tabla)