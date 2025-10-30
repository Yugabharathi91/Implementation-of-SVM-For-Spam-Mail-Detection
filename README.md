# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Input: Load and read the spam dataset containing messages and their labels.

2.Processing: Clean the text, remove stopwords, and convert it into TF-IDF feature vectors.

3.Decision: Train the SVM classifier and use it to predict whether a message is spam or not.

4.Output: Display results such as accuracy, confusion matrix, and message prediction.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: S.Dharshini
RegisterNumber: 212224230061 
*/
```
```


import pandas as pd
import numpy as np
import re
import string
import joblib
import nltk
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve
)

url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
data = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

print(" Dataset Loaded Successfully!")
print("Total Messages:", len(data))
print(data.head())

def clean_text(text):
    text = text.lower()  # lowercase
    text = re.sub(r"http\S+", "", text)  # remove URLs
    text = re.sub(r"\S+@\S+", "", text)  # remove emails
    text = re.sub(r"\d+", "", text)  # remove numbers
    text = text.translate(str.maketrans("", "", string.punctuation))  # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()  # remove extra spaces
    return text

data['clean_msg'] = data['message'].apply(clean_text)

X = data['clean_msg']
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

vectorizer = TfidfVectorizer(
    stop_words=stopwords.words('english'),
    ngram_range=(1, 2),
    max_df=0.9,
    min_df=2,
    max_features=5000
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

best_svm = LinearSVC(C=1.0, max_iter=5000)
best_svm.fit(X_train_tfidf, y_train)
print("\n SVM Model Trained Successfully (Linear Kernel)")

y_pred = best_svm.predict(X_test_tfidf)

print("\n Evaluation Metrics:")
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=4))

y_scores = best_svm.decision_function(X_test_tfidf)
fpr, tpr, _ = roc_curve(y_test, y_scores)
roc_auc = roc_auc_score(y_test, y_scores)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()

joblib.dump(best_svm, "svm_spam_model.joblib")
joblib.dump(vectorizer, "tfidf_vectorizer.joblib")
print("\n Model and Vectorizer Saved Successfully!")

def predict_message(msg):
    msg_clean = clean_text(msg)
    msg_vec = vectorizer.transform([msg_clean])
    pred = best_svm.predict(msg_vec)[0]
    print(f"\nMessage: {msg}")
    print("Prediction:", " SPAM" if pred == 1 else " HAM (Not Spam)")

predict_message("Congratulations! You won a $1000 gift card. Call now!")
predict_message("Hey, are we still on for the meeting tomorrow?")



```


## Output:

<img width="923" height="436" alt="image" src="https://github.com/user-attachments/assets/d188f792-5f59-4b32-9e5a-1adc0611969c" />

<img width="870" height="261" alt="image" src="https://github.com/user-attachments/assets/8138f34a-1eae-4798-9a01-602487579f1f" />

<img width="1047" height="733" alt="image" src="https://github.com/user-attachments/assets/1ad4c698-addd-4405-b06b-37974afefe48" />


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
