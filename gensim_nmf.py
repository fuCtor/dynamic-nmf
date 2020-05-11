import text.util
(X,terms,doc_ids,tfids, docs) = text.util.load_corpus( "data/month3.pkl" )

from gensim.corpora.dictionary import Dictionary
from gensim.models.nmf import Nmf
from gensim.models import CoherenceModel
from prettytable import PrettyTable

import itertools
import networkx as nx
import matplotlib.pyplot as plt
    
x = PrettyTable()

common_dictionary = Dictionary(docs)
common_corpus = [common_dictionary.doc2bow(text) for text in docs]

# for k in range(4, 10):
#     nmf = Nmf(common_corpus, num_topics=k)
#     c_model = CoherenceModel(model=nmf, corpus=common_corpus, dictionary=common_dictionary, texts=docs, coherence='c_v')
#     print(k, c_model.get_coherence())
#     x = PrettyTable()
#     x.field_names = [''] + [ "t" + str(i+1) for i in range(0,10)]
#     for i in range(0,k):
#         x.add_row([i] + [ common_dictionary[term] for (term, w)  in nmf.get_topic_terms(i)])
#     print(x)

from gensim.matutils import jaccard
import random 
nmf = Nmf(common_corpus, num_topics=9)

texts = random.choices(docs, k=20)
texts = [docs[0], docs[20], docs[80], docs[90], docs[200], docs[210]] #[docs[i] for i in range(0, len(docs), 30)]

def get_most_likely_topic(doc):
    bow = common_dictionary.doc2bow(doc)
    topics, probabilities = zip(*nmf.get_document_topics(bow))
    max_p = max(probabilities)
    topic = topics[probabilities.index(max_p)]
    return topic

colors =  ["skyblue", "pink", "red", "green", "yellow", "cyan", "purple", "magenta", "orange", "blue"]
def get_node_color(i):
    return colors[get_most_likely_topic(texts[i])]
    # return 'skyblue' if get_most_likely_topic(texts[i]) == 0 else 'pink'

G = nx.Graph()
for i, _ in enumerate(texts):
    G.add_node(i)
    
for (i1, i2) in itertools.combinations(range(len(texts)), 2):
    bow1, bow2 = texts[i1], texts[i2]
    distance = jaccard(bow1, bow2)
    if(distance > 0.001):
        G.add_edge(i1, i2, weight=1/distance) 

pos = nx.spring_layout(G)

threshold = 1.04
elarge=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] > threshold]
esmall=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] <= threshold]

node_colors = [get_node_color(i) for (i, _) in enumerate(texts)]
nx.draw_networkx_nodes(G, pos, node_size=700, node_color=node_colors)
nx.draw_networkx_edges(G,pos,edgelist=elarge, width=2)
nx.draw_networkx_edges(G,pos,edgelist=esmall, width=2, alpha=0.2, edge_color='b', style='dashed')
nx.draw_networkx_labels(G, pos, font_size=20, font_family='sans-serif')
plt.show()