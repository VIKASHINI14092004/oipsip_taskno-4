# oipsip_taskno-4 Spam Message Classification
# Introduction
Spam messages, also known as unsolicited messages, are a common nuisance in today's digital communication channels. They often contain unwanted advertisements, scams, or malicious content. Automatic spam detection systems are essential for filtering out such messages and ensuring a better user experience. In this project, we develop a spam message classification model using Natural Language Processing (NLP) techniques and machine learning algorithms.

# Objective
The primary objective of this project is to build a robust spam message classification model that can accurately identify whether a given message is spam or not. The model will be trained on a dataset containing labeled examples of spam and non-spam (ham) messages. We aim to achieve high classification performance in terms of precision, recall, and F1-score.

# Methodology
1. Data Collection
We obtained a dataset of SMS messages labeled as spam or ham from an online repository. The dataset consists of text messages along with their corresponding labels.

2. Data Preprocessing
Text Cleaning: We performed various preprocessing steps on the text data, including converting text to lowercase, removing punctuation, and eliminating stopwords.
Vectorization: We utilized the Term Frequency-Inverse Document Frequency (TF-IDF) technique to convert text data into numerical features.
3. Model Development
We trained a Multinomial Naive Bayes classifier on the preprocessed and vectorized text data. Naive Bayes is a popular choice for text classification tasks due to its simplicity and effectiveness.

4. Model Evaluation
We evaluated the performance of the trained model using standard classification metrics such as confusion matrix, precision, recall, and F1-score. The evaluation was conducted on a separate test set that was not seen by the model during training.

