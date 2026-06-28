import sys
import os

sys.path.append(os.path.abspath(".."))


import pandas as pd
import string
import joblib
from src.preprocess import preprocess_text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix,precision_score,recall_score,f1_score,classification_report

def main():
    

    data = pd.read_csv("../data/email.csv", encoding="utf-8")
    df = pd.DataFrame(data)


    df['processed_Message'] = df['Message'].apply(preprocess_text)
    df['clean_message'] = df['processed_Message'].str.join(" ")


    cv = CountVectorizer()

    X = cv.fit_transform(df["clean_message"])

    X_train,X_test,y_train,y_test = train_test_split(X,df['Category'],test_size=0.2,random_state=42)

    naive_model = MultinomialNB()
    naive_model.fit(X_train,y_train)
    y_pred = naive_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy*100:.2f}%")

    cm = confusion_matrix(y_test,y_pred)

    precision = precision_score(y_test,y_pred,pos_label="spam")
    recall = recall_score(y_test,y_pred,pos_label="spam")
    f1 = f1_score(y_test,y_pred,pos_label="spam")

    print(cm)
    print(precision)
    print(recall)
    print(f1)

    cd = classification_report(y_test,y_pred)
    print(cd)




    joblib.dump(naive_model,"../models/spam_model.joblib")
    joblib.dump(cv,"../models/vectorizer.joblib")

if __name__ == "__main__":
    main()