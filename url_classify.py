from gensim import corpora, models, similarities
import glob
from scipy.sparse import csr_matrix
import pickle
from sklearn.model_selection import train_test_split
dirs = glob.glob('urls/*')
f = open('stop-words-english.txt', 'r')
stoplist = [w.strip() for w in f.readlines() if w]
label = 0
documents = []
doc_Y = []
for x in dirs:
    print x
    for fl in glob.glob(x+'/*.plain'):
	doc_Y.append(label)
	with open(fl, 'rb') as f:
	    documents.append(unicode(f.read(), errors='replace'))
    label = label+1

texts = [[word for word in document.lower().split() if word not in stoplist]
         for document in documents]
dictionary = corpora.Dictionary(texts)
small_freq_ids = [tokenid for tokenid, docfreq in dictionary.dfs.items() if docfreq < 5 ]
dictionary.filter_tokens(small_freq_ids)
dictionary.compactify()
dictionary.save("url.dict")

corpus = [dictionary.doc2bow(text) for text in texts]
tfidf = models.TfidfModel(corpus)
with open('tfidf.pickle', 'wb') as f:
    pickle.dump(tfidf, f)

from sklearn import svm
corpus_tfidf = tfidf[corpus]
for topic in [200,250,300,350,400]:
    print "TOPIC",topic
    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=topic) # initialize an LSI transformation
    #with open('lsi.pickle', 'wb') as f:
    #	pickle.dump(lsi, f)
    corpus_lsi = lsi[corpus_tfidf] #
    data = []
    rows = []
    cols = []
    line_count = 0
    for line in corpus_lsi:
    	for elem in line:
	    rows.append(line_count)
	    cols.append(elem[0])
	    data.append(elem[1])
    	line_count += 1
    lsi_matrix = csr_matrix((data,(rows,cols))).toarray()

    X_train, X_test, y_train, y_test = train_test_split(lsi_matrix, doc_Y, test_size=0.3, random_state=0)
    clf = svm.LinearSVC()
    clf.fit(X_train,y_train)
    y_pre = clf.predict(X_test)
    total = 0
    error = 0
    for (y,y_pre) in zip(y_test,y_pre):
    	if y!=y_pre:
	    error = error+1
    	total = total+1
    print total,error
with open('lsi.pickle', 'wb') as f:
    pickle.dump(lsi, f)

"""
lda = models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=2)
ldaout = lda.print_topics(5)
for x in ldaout:
    print x
"""