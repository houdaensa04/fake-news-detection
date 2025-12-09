import pandas as pd
import numpy as nm
from sklearn.model_selection import train_test_split as ttp
from sklearn.metrics import accuracy_score, classification_report
import re
import string
import matplotlib.pyplot as plt

data_true=pd.read_csv("True.csv")
data_fake=pd.read_csv("Fake.csv")

data_true["class"]=1
data_fake["class"]=0

data_true_manual_testing = data_true.tail(10)
for i in range(21416, 21406, -1):
    data_true.drop([i], axis=0, inplace=True)  # removing those 10 values from original dataset

data_fake_manual_testing = data_fake.tail(10)
for i in range(21416, 21406, -1):
    data_fake.drop([i], axis=0, inplace=True)  # removing those 10 values from original dataset

data_manual_testing = pd.concat([data_fake_manual_testing, data_true_manual_testing], axis=0)
data_manual_testing.to_csv("manual_testing.csv ")

data_merge = pd.concat([data_fake,data_true],axis=0)

data = data_merge.drop(["title","subject","date"], axis=1)
data=data.sample(frac=1)

def filtering(data):
    text=data.lower()
    text=re.sub(r'\[.*?\]','',text)
    text=re.sub(r"\\W"," ",text)
    text=re.sub(r'https?://\s+|www\.S+','',text)
    text=re.sub(r'<.*?>+','',text)
    text=re.sub(r'[%s]'% re.escape(string.punctuation),'',text)
    text=re.sub(r'\w*\d\w*','',text)
    return text

data["text"]= data["text"].apply(filtering)

x=data["text"]
y=data["class"]
x_train,x_test,y_train,y_test=ttp(x,y,test_size=0.25,random_state=0)

from sklearn.feature_extraction.text import TfidfVectorizer
vector = TfidfVectorizer()
xv_train=vector.fit_transform(x_train)
xv_test=vector.transform(x_test)

from sklearn.linear_model import LogisticRegression
LR=LogisticRegression()
LR.fit(xv_train,y_train)
pred= LR.predict(xv_test)
print("Accuracy :", accuracy_score(y_test, pred))
print("\nReport :\n", classification_report(y_test, pred))

from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier()
DT.fit(xv_train,y_train)
pred_DT = DT.predict(xv_test)
print(classification_report(y_test,pred_DT))

from sklearn.ensemble import GradientBoostingClassifier
GBC = GradientBoostingClassifier(random_state=0)
GBC.fit(xv_train,y_train)
pred_GBC = GBC.predict(xv_test)
print(classification_report(y_test,pred_GBC))

from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier(random_state=0)
RFC.fit(xv_train,y_train)
pred_RFC = RFC.predict(xv_test)
print(classification_report(y_test,pred_RFC))


def output_lable(n):
    if n == 0:
        return "FAKE News"
    elif n == 1:
        return "TRUE News"


def manual_testing(news):
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(filtering)
    new_x_test = new_def_test["text"]
    new_xv_test = vector.transform(new_x_test)
    pred_LR = LR.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)
    pred_GBC = GBC.predict(new_xv_test)
    pred_RFC = RFC.predict(new_xv_test)

    result = (
        f"LR Prediction: {output_lable(pred_LR[0])}\n"
        f"DT Prediction: {output_lable(pred_DT[0])}\n"
        f"GBC Prediction: {output_lable(pred_GBC[0])}\n"
        f"RFC Prediction: {output_lable(pred_RFC[0])}"
    )

    return result



# news = input("\nÉcris une news à tester : ")
# manual_testing(news)




