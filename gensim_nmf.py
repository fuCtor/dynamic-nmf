import text.util
(X,terms,doc_ids,tfids, docs) = text.util.load_corpus( "data/month3.pkl" )

from gensim.corpora.dictionary import Dictionary
from gensim.models.nmf import Nmf
from gensim.models import CoherenceModel
from prettytable import PrettyTable
    
x = PrettyTable()

common_dictionary = Dictionary(docs)
common_corpus = [common_dictionary.doc2bow(text) for text in docs]

for k in range(4, 10):
    nmf = Nmf(common_corpus, num_topics=k)
    c_model = CoherenceModel(model=nmf, corpus=common_corpus, dictionary=common_dictionary, texts=docs, coherence='c_v')
    print(k, c_model.get_coherence())
    x = PrettyTable()
    x.field_names = [''] + [ "t" + str(i+1) for i in range(0,10)]
    for i in range(0,k):
        x.add_row([i] + [ common_dictionary[term] for (term, w)  in nmf.get_topic_terms(i)])
    print(x)