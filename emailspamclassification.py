import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from nltk.corpus import stopwords
import string
import nltk

# Load the dataset
df = pd.read_csv('spam.csv', encoding='Windows-1252')

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()  # convert to lowercase
    text = ''.join([char for char in text if char not in string.punctuation])  # remove punctuation
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])  # remove stopwords
    return text

# Preprocess text data
df['text'] = df['text'].apply(preprocess_text)

# Vectorizing text
tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(df['text'])
y = df['label']

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predicting the test set
y_pred = model.predict(X_test)

# Evaluating the model
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print('Confusion Matrix:\n', conf_matrix)
print('\nClassification Report:\n', report)
