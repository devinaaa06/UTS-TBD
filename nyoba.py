#!/usr/bin/env python

import pandas as pd
import sklearn.metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Baca dataset
df = pd.read_csv('diabetes.csv')

def mapper(row):
    # Inisialisasi fitur (features) dan label
    X = row['Age']  # Hanya menggunakan kolom 'Age' sebagai fitur
    y = row['Outcome']

    # Buat dan latih model Random Forest
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit([[X]], [y])

    # Prediksi dengan model yang telah dilatih
    prediction = clf.predict([[X]])

    return prediction

def reducer(data):
    total_count = len(data)
    diabetes_count = sum(data)

    return pd.Series({'Jumlah pada umur': total_count, 'Jumlah orang yang terkena diabetes': diabetes_count})

# Apply MapReduce
df['diabetes_pred'] = df.apply(mapper, axis=1)
df_diabetes = df.groupby('Age')['diabetes_pred'].apply(reducer).reset_index()

# Split dataset ke dataset training dan testing
X_train, X_test, y_train, y_test = train_test_split(df[['Age']], df['Outcome'], test_size=0.3, random_state=42)

# Latih model dengan dataset training
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluasi model dengan dataset testing
y_pred = clf.predict(X_test)
accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')

print(df_diabetes)