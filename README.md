# spam-email-classifier

# Import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk # Natural Language Toolkit
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB

# Load data & print samples
df = pd.read_csv('/kaggle/input/email-spam-detection-dataset-classification/spam.csv', encoding='latin-1')
df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
df = df.rename(columns={"v1": "Category", "v2": "Text"})
print(df.head())

# Dataset info
print("Total number of rows in the dataset are", len(df))
print(df.describe())

# Add text length column
df['Length'] = df['Text'].apply(len)
print(df.head())

# Plot distribution of categories
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
fig, ax = plt.subplots(1, 2, figsize=(15, 4))
ax = ax.flatten()
value_counts = df['Category'].value_counts()
labels = value_counts.index.tolist()
colors = ["#6782a8", "#ab90a0"]
wedges, texts, autotexts = ax[0].pie(
    value_counts, autopct='%1.1f%%', textprops={'size': 9, 'color': 'white', 'fontweight': 'bold'}, colors=colors,
    wedgeprops=dict(width=0.35), startangle=80, pctdistance=0.85)
centre_circle = plt.Circle((0, 0), 0.6, fc='white')
ax[0].add_artist(centre_circle)
sns.countplot(data=df, y=df['Category'], ax=ax[1], palette=colors, order=labels)
for i, v in enumerate(value_counts):
    ax[1].text(v + 1, i, str(v), color='black', fontsize=10, va='center')
sns.despine(left=True, bottom=True)
plt.yticks(fontsize=9, color='black')
ax[1].set_ylabel(None)
plt.xlabel("")
plt.xticks([])
fig.suptitle('Spam - Ham Distribution', fontsize=15)
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.show()

# Histogram of text length
import plotly.express as px
fig = px.histogram(df, x='Length', marginal='rug', title='Histogram of Text Length')
fig.update_layout(xaxis_title='Length', yaxis_title='Frequency', showlegend=True)
fig.show()

# Histogram of text length by category
fig = px.histogram(df, x='Length', color='Category', marginal='rug', title='Histogram of Text Length by Category')
fig.update_layout(xaxis_title='Length', yaxis_title='Frequency', showlegend=True)
fig.show()

# Label encoding: Spam as 1, Ham as 0
df['Category'] = df.Category.map({'ham': 0, 'spam': 1}).astype(int)
print(df.head())

# (Optional) Install wordcloud if not already installed
# !pip install wordcloud

# WordClouds for Spam and Ham
from wordcloud import WordCloud, STOPWORDS
spam = df[df['Category'] == 1]
ham = df[df['Category'] == 0]
# font_path = "/kaggle/input/fonts/acetone_font.otf" # Uncomment if you have a custom font

def generate_wordcloud(data, title):
    words = ' '.join(data['Text'])
    wordcloud = WordCloud(stopwords=STOPWORDS,
                          # font_path=font_path, # Uncomment if you have a custom font
                          max_words=1500,
                          max_font_size=350, random_state=42,
                          width=2000, height=800,
                          colormap="twilight").generate(words)
    plt.figure(figsize=(16, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(title)
    plt.show()

generate_wordcloud(spam, 'Spam WordCloud')
generate_wordcloud(ham, 'Ham WordCloud')

# Bag of Words feature extraction
count = CountVectorizer()
text = count.fit_transform(df['Text'])
x_train, x_test, y_train, y_test = train_test_split(text, df['Category'], test_size=0.30, random_state=100)
print('X-Train :', x_train.shape)
print('X-Test :', x_test.shape)
print('Y-Train :', y_train.shape)
print('Y-Test :', y_test.shape)

# Train MLP Classifier
mlp_classifier_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000)
mlp_classifier_model.fit(x_train, y_train)
prediction = mlp_classifier_model.predict(x_test)
print("MLP Classifier")
print("Accuracy score: {:.2f}".format(accuracy_score(y_test, prediction)))
print("Precision score: {:.2f}".format(precision_score(y_test, prediction)))
print("Recall score: {:.2f}".format(recall_score(y_test, prediction)))
print("F1 score: {:.2f}".format(f1_score(y_test, prediction)))

# Train Multinomial Naive Bayes
multinomial_nb_model = MultinomialNB()
multinomial_nb_model.fit(x_train, y_train)
prediction = multinomial_nb_model.predict(x_test)
print("Multinomial NB")
print("Accuracy score: {:.2f}".format(accuracy_score(y_test, prediction)))
print("Precision score: {:.2f}".format(precision_score(y_test, prediction)))
print("Recall score: {:.2f}".format(recall_score(y_test, prediction)))
print("F1 score: {:.2f}".format(f1_score(y_test, prediction)))

# Train Bernoulli Naive Bayes
bernoulli_nb_model = BernoulliNB()
bernoulli_nb_model.fit(x_train, y_train)
prediction = bernoulli_nb_model.predict(x_test)
print("Bernoulli NB")
print("Accuracy score: {:.2f}".format(accuracy_score(y_test, prediction)))
print("Precision score: {:.2f}".format(precision_score(y_test, prediction)))
print("Recall score: {:.2f}".format(recall_score(y_test, prediction)))
print("F1 score: {:.2f}".format(f1_score(y_test, prediction)))

# Confusion Matrix Subplot for 3 Models
models = [("Multinomial NB", multinomial_nb_model), ("Bernoulli NB", bernoulli_nb_model), ("MLP Classifier", mlp_classifier_model)]
fig, axes = plt.subplots(1, 3, figsize=(10, 3))
for i, (model_name, model) in enumerate(models):
    prediction = model.predict(x_test)
    cm = confusion_matrix(y_test, prediction)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[i])
    axes[i].set_title(f"{model_name} - Confusion Matrix")
    axes[i].set_xlabel("Predicted")
    axes[i].set_ylabel("Actual")
plt.tight_layout()
plt.show()

# Metric Comparison Heatmap
metric_data = []
for model_name, model in models:
    prediction = model.predict(x_test)
    accuracy = accuracy_score(y_test, prediction)
    precision = precision_score(y_test, prediction)
    recall = recall_score(y_test, prediction)
    f1 = f1_score(y_test, prediction)
    metric_data.append([accuracy, precision, recall, f1])
metric_labels = ["Accuracy", "Precision", "Recall", "F1 Score"]
plt.figure(figsize=(6, 3))
sns.heatmap(metric_data, annot=True, fmt=".2f", cbar=False, cmap="summer_r",
            xticklabels=metric_labels,
            yticklabels=[model_name for model_name, _ in models])
plt.title("Metric Comparison")
plt.yticks(rotation=0)
plt.xlabel("Metrics")
plt.ylabel("Models")
plt.tight_layout()
plt.show()
