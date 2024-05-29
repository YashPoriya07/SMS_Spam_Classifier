#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df=pd.read_csv('spam.csv')


# In[3]:


df.head()


# In[4]:


df.shape


# ### Steps :-
# 1. Data Cleaning
# 2. EDA
# 3. Text Pre processing
# 4. Model Building
# 5. Evaluation
# 6. Improvements
# 7. Website

# ## Data Cleaning

# In[5]:


df.info()


# In[6]:


#dropping last 3 columns
df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)


# In[7]:


df.head()


# In[8]:


#Renaming the columns
df.rename(columns={'v1':'target','v2':'text'},inplace=True)


# In[9]:


df.head()


# In[10]:


from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()


# In[11]:


df['target']=encoder.fit_transform(df['target'])


# In[12]:


df.head()


# In[13]:


df.isnull().sum()


# In[14]:


#check for duplicate values
df.duplicated().sum()


# In[15]:


df=df.drop_duplicates(keep='first')


# In[16]:


df.duplicated().sum()


# In[17]:


df.shape


# ## E D A

# In[18]:


df.head()


# In[19]:


df['target'].value_counts()


# In[20]:


import matplotlib.pyplot as plt
plt.pie(df['target'].value_counts(),labels=['ham','spam'],autopct="%0.2f")
plt.show()


# #### DATA IS IMBALANCED

# In[21]:


import nltk
#natural language tool kit


# In[22]:


nltk.download('punkt') #downloading the dependencies for nltk


# In[23]:


df['num_characters']=df['text'].apply(len) #har sms ka lenght dega w.r.t to characters


# In[24]:


df.head()


# In[25]:


#fetching number of words
df['text'].apply(lambda x:nltk.word_tokenize(x)) #breaking the sms into words


# In[26]:


df['num_words']=df['text'].apply(lambda x:len(nltk.word_tokenize(x)))


# In[27]:


df.head()


# In[28]:


df['text'].apply(lambda x:nltk.sent_tokenize(x)) #breaking the sms on the basis of sentences


# In[29]:


df['num_sentences']=df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))


# In[30]:


df.head()


# In[31]:


df[['num_characters','num_words','num_sentences']].describe()


# In[32]:


#ham messages
df[df['target']==0][['num_characters','num_words','num_sentences']].describe()


# In[33]:


#spam messages
df[df['target']==1][['num_characters','num_words','num_sentences']].describe()


# In[34]:


import seaborn as sns


# In[35]:


sns.histplot(df[df['target']==0]['num_characters'])
sns.histplot(df[df['target']==1]['num_characters'],color='red')


# In[36]:


sns.pairplot(df,hue='target')


# In[37]:


df.info()


# In[38]:


print(df.dtypes)


# In[39]:


sns.heatmap(df.drop(columns='text').corr(),annot=True)


# ## Data Pre processing
# 1. Lower Case
# 2. Tokenization
# 3. Removing special characters 
# 4. Removing stop words and punctuations
# 5. Stemming

# In[40]:


from nltk.corpus import stopwords
stopwords.words('english')


# In[41]:


import string
string.punctuation


# In[42]:


from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


# In[43]:


def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text) #removing special characters
    
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
        
    text=y[:] #clonning
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    
    text=y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
        
    return " ".join(y)


# In[44]:


#for example
transform_text('i loved the YT LECTURES on Machine Learning. How about you?')


# In[45]:


df['text'][0]


# In[46]:


df['text'].apply(transform_text)


# In[47]:


df['transform_text']=df['text'].apply(transform_text)


# In[48]:


df.head()


# In[49]:


from wordcloud import WordCloud
wc=WordCloud(width=500,height=500,min_font_size=10,background_color='white')


# In[50]:


spam_wc=wc.generate(df[df['target']==1]['transform_text'].str.cat(sep=" "))


# In[51]:


plt.figure(figsize=(15,6))
plt.imshow(spam_wc)


# In[52]:


ham_wc=wc.generate(df[df['target']==0]['transform_text'].str.cat(sep=" "))


# In[53]:


plt.figure(figsize=(15,6))
plt.imshow(ham_wc)


# In[54]:


df.head()


# In[55]:


spam_corpus=[]
for msg in df[df['target']==1]['transform_text'].tolist():
    for word in msg.split():
        spam_corpus.append(word)


# In[56]:


len(spam_corpus)


# In[57]:


from collections import Counter
Counter(spam_corpus).most_common(30)


# In[58]:


pd.DataFrame(Counter(spam_corpus).most_common(30))


# In[59]:


# top 30 words used in spam sms
sns.barplot(x=pd.DataFrame(Counter(spam_corpus).most_common(30))[0],y=pd.DataFrame(Counter(spam_corpus).most_common(30))[1])
plt.xticks(rotation='vertical')
plt.show()


# In[60]:


ham_corpus=[]
for msg in df[df['target']==0]['transform_text'].tolist():
    for word in msg.split():
        ham_corpus.append(word)


# In[61]:


len(ham_corpus)


# In[62]:


# top 30 words used in ham sms
sns.barplot(x=pd.DataFrame(Counter(ham_corpus).most_common(30))[0],y=pd.DataFrame(Counter(ham_corpus).most_common(30))[1])
plt.xticks(rotation='vertical')
plt.show()


# ## Model Building

# In[63]:


df.head()


# In[64]:


from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
tfidf=TfidfVectorizer(max_features=3000)
cv=CountVectorizer()


# In[65]:


x=cv.fit_transform(df['transform_text']).toarray() #converting sparse array to dense array
X=tfidf.fit_transform(df['transform_text']).toarray() 


# In[66]:


x


# In[67]:


X


# In[68]:


x.shape


# In[69]:


X.shape


# In[70]:


y=df['target'].values


# In[71]:


y


# In[72]:


from sklearn.model_selection import train_test_split


# In[73]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=2)


# In[74]:


from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score


# In[75]:


gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()


# In[76]:


#CountVectorizer
gnb.fit(x_train,y_train)
y_pred1=gnb.predict(x_test)
print(accuracy_score(y_test,y_pred1))
print()
print(confusion_matrix(y_test,y_pred1))
print()
print(precision_score(y_test,y_pred1))


# In[77]:


#TfidfVectorizer
gnb.fit(X_train,y_train)
y_pred1=gnb.predict(X_test)
print(accuracy_score(y_test,y_pred1))
print()
print(confusion_matrix(y_test,y_pred1))
print()
print(precision_score(y_test,y_pred1))


# In[78]:


#CountVectorizer
mnb.fit(x_train,y_train)
y_pred2=mnb.predict(x_test)
print(accuracy_score(y_test,y_pred2))
print()
print(confusion_matrix(y_test,y_pred2))
print()
print(precision_score(y_test,y_pred2))


# In[79]:


#TfidfVectorizer
mnb.fit(X_train,y_train)
y_pred2=mnb.predict(X_test)
print(accuracy_score(y_test,y_pred2))
print()
print(confusion_matrix(y_test,y_pred2))
print()
print(precision_score(y_test,y_pred2))


# In[80]:


#CountVectorizer
bnb.fit(x_train,y_train)
y_pred3=bnb.predict(x_test)
print(accuracy_score(y_test,y_pred3))
print()
print(confusion_matrix(y_test,y_pred3))
print()
print(precision_score(y_test,y_pred3))


# In[81]:


#TfidfVectorizer
bnb.fit(X_train,y_train)
y_pred3=bnb.predict(X_test)
print(accuracy_score(y_test,y_pred3))
print()
print(confusion_matrix(y_test,y_pred3))
print()
print(precision_score(y_test,y_pred3))


# In[82]:


# since data is imbalanced precision score is more important than accuracy
#therefor we will use MultinomialNB (tfidf)


# In[83]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier


# In[84]:


svc = SVC(kernel='sigmoid', gamma=1.0)
mnb = MultinomialNB()
dtc = DecisionTreeClassifier(max_depth=5)
lrc = LogisticRegression(solver='liblinear', penalty='l1')
rfc = RandomForestClassifier(n_estimators=50, random_state=2)
abc = AdaBoostClassifier(n_estimators=50, random_state=2)
bc = BaggingClassifier(n_estimators=50, random_state=2)
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)
gbdt = GradientBoostingClassifier(n_estimators=50,random_state=2)
xgb = XGBClassifier(n_estimators=50,random_state=2)


# In[85]:


clfs = {
    'SVC' : svc, 
    'NB': mnb, 
    'DT': dtc, 
    'LR': lrc, 
    'RF': rfc, 
    'AdaBoost': abc, 
    'BgC': bc, 
    'ETC': etc,
    'GBDT':gbdt,
    'xgb':xgb
}


# In[86]:


def train_classifier(clf,X_train,y_train,X_test,y_test):
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    
    return accuracy,precision


# In[87]:


#for example
train_classifier(svc,X_train,y_train,X_test,y_test)


# In[88]:


accuracy_scores = []
precision_scores = []

for name,clf in clfs.items():
    
    current_accuracy,current_precision = train_classifier(clf, X_train,y_train,X_test,y_test)
    
    print("For ",name)
    print("Accuracy - ",current_accuracy)
    print("Precision - ",current_precision)
    print()
    
    accuracy_scores.append(current_accuracy)
    precision_scores.append(current_precision)


# In[89]:


performance_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy':accuracy_scores,'Precision':precision_scores}).sort_values('Precision',ascending=False)


# In[90]:


performance_df


# In[91]:


performance_df1 = pd.melt(performance_df, id_vars = "Algorithm")


# In[92]:


performance_df1


# In[93]:


sns.catplot(x = 'Algorithm', y='value', 
               hue = 'variable',data=performance_df1, kind='bar',height=5)
plt.ylim(0.5,1.0)
plt.xticks(rotation='vertical')
plt.show()


# In[94]:


# Model Improvement
# 1. Changed the max_features parameter of TfIdf


# In[95]:


# Voting Classifier -> combination of best working models
svc = SVC(kernel='sigmoid', gamma=1.0,probability=True)
mnb = MultinomialNB()
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)

from sklearn.ensemble import VotingClassifier


# In[96]:


voting = VotingClassifier(estimators=[('svm', svc), ('nb', mnb), ('et', etc)],voting='soft')


# In[97]:


voting.fit(X_train,y_train)


# In[98]:


y_pred = voting.predict(X_test)
print("Accuracy",accuracy_score(y_test,y_pred))
print("Precision",precision_score(y_test,y_pred))


# In[99]:


# Applying stacking -> combining of best working models and giving them weights/stacking
estimators=[('svm', svc), ('nb', mnb), ('et', etc)]
final_estimator=RandomForestClassifier()


# In[100]:


from sklearn.ensemble import StackingClassifier
clf = StackingClassifier(estimators=estimators, final_estimator=final_estimator)


# In[101]:


clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy",accuracy_score(y_test,y_pred))
print("Precision",precision_score(y_test,y_pred))


# In[102]:


#using VotingClassifier and Stacking kuch zyada fark nahi pada
#so now we will use MultinomialNB itself


# In[103]:


import pickle
pickle.dump(tfidf,open('vectorizer.pkl','wb'))
pickle.dump(mnb,open('model.pkl','wb'))


# In[ ]:




