import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import urllib.request

# Function to download the dataset
def download_dataset(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)
        print("Download completed.")
    else:
        print(f"{filename} already exists.")

# URL of the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
zip_file = "smsspamcollection.zip"
csv_file = "SMSSpamCollection"

# Download and extract the dataset
download_dataset(url, zip_file)

# Unzip the file
import zipfile
with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_ref.extractall()

# Load the dataset
df = pd.read_csv(csv_file, sep='\t', names=['label', 'text'], encoding='latin-1')

# The rest of the code remains the same
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Create a Bag of Words model
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

# Train a Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train_counts, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test_counts)

# Print the classification report
print(classification_report(y_test, y_pred))

# Create a confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Function to classify new SMS
def classify_sms(sms_text):
    sms_counts = vectorizer.transform([sms_text])
    prediction = clf.predict(sms_counts)
    return "Spam" if prediction[0] == 1 else "Not Spam"

# Test the classifier with some example SMS
print(classify_sms("Congratulations! You've won a free iPhone. Click here to claim your prize!"))
print(classify_sms("Hey, what time are we meeting for dinner tonight?"))