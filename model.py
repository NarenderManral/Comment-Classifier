import sklearn.ensemble
import sklearn.naive_bayes
import sklearn.model_selection
import pandas as pd
import numpy as np
import nltk.stem.porter
import nltk.corpus
import sklearn.feature_extraction
X_train=pd.read_csv("C:\\Users\\manral\\Desktop\\ML\\train.csv\\train.csv")
X2=(X_train["comment_text"])
Y=X_train["severe_toxic"]
porter_stemmer=nltk.stem.porter.PorterStemmer()

stop_words=set(nltk.corpus.stopwords.words("english"))
X=[]
for sentence in X2[1:1000]:
    sentence=sentence.lower()
    words=sentence.split(" ")
    final_sentence=[]
    new_sentence=""
    for word in words:
        if word not in stop_words:
            word_stem=porter_stemmer.stem(word)
            final_sentence.append(word_stem)
    new_sentence=" ".join(final_sentence)
    X.append(new_sentence)
    
vect=sklearn.feature_extraction.text.CountVectorizer()
tf_train=vect.fit_transform(X)
x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(tf_train,Y[1:1000],test_size=0.2)
model=sklearn.naive_bayes.GaussianNB()
model.fit(x_train.toarray(),y_train)
print(model.score(x_test.toarray(),y_test))
