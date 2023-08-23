import random
import re
import string

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from tabulate import tabulate


def preprocess_text(text):
    if isinstance(text, str):
        text = re.sub(f"[{string.punctuation}]|[0-9]", "", text)
        text = text.lower()

        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]

        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]

        preprocessed_text = ' '.join(lemmatized_tokens)
        return preprocessed_text


def predict_category(classifier, vectorizer):
    title = input("Enter the title: ")
    abstract = input("Enter the abstract: ")

    combined_text = title + ' ' + abstract
    preprocessed_input = preprocess_text(combined_text)
    input_vec = vectorizer.transform([preprocessed_input])
    predicted_category = classifier.predict(input_vec)
    print(f"Predicted category: {predicted_category[0]}")


def display_some_predictions(predictions, actuals, texts):
    num_predictions_to_display = 20
    random_indices = random.sample(range(len(predictions)), num_predictions_to_display)
    display_data = []

    for i in random_indices:
        display_data.append([predictions[i], actuals.iloc[i], texts.iloc[i]])

    headers = ["Predicted", "Actual", "Text"]
    print("\nSome Predictions:")
    print(tabulate(display_data, headers=headers, tablefmt="grid"))


def plot_confusion_matrix(y_test, y_pred, categories):
    cm = confusion_matrix(y_test, y_pred, labels=categories)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=categories, yticklabels=categories)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.show()


def plot_precision_recall_curves(y_test, y_pred, categories):
    precision, recall, average_precision = dict(), dict(), dict()
    for i, cat in enumerate(categories):
        precision[cat], recall[cat], _ = precision_recall_curve(y_test == cat, y_pred == cat)
        average_precision[cat] = auc(recall[cat], precision[cat])

    plt.figure(figsize=(10, 8))
    for cat in categories:
        plt.plot(recall[cat], precision[cat], lw=2, label=f'{cat} (AP = {average_precision[cat]:.2f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def run_naive_bayes_with_ngrams(ngram_range):
    dataset = pd.read_csv('articles.csv', delimiter='|')
    dataset['text'] = dataset['title'] + ' ' + dataset['abstract']

    dataset.dropna(subset=['text', 'category'], inplace=True)
    dataset['preprocessed_text'] = dataset['text'].apply(preprocess_text)

    x_train, x_test, y_train, y_test = train_test_split(dataset['preprocessed_text'], dataset['category'],
                                                        stratify=dataset['category'], test_size=0.2, random_state=42)

    vectorizer = CountVectorizer(ngram_range=ngram_range)
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)

    oversampler = RandomOverSampler(random_state=42)
    x_train_vec_resampled, y_train_resampled = oversampler.fit_resample(x_train_vec, y_train)

    classifier = MultinomialNB()
    classifier.fit(x_train_vec_resampled, y_train_resampled)

    y_pred = classifier.predict(x_test_vec)

    categories = dataset['category'].unique()
    num_unique_classes = len(y_test.unique())
    class_report = classification_report(y_test, y_pred, target_names=categories[:num_unique_classes],
                                         zero_division='warn')

    print(class_report)
    plot_confusion_matrix(y_test, y_pred, categories)
    plot_precision_recall_curves(y_test, y_pred, categories)


    while True:
        print("\nMenu:")
        print("1 - Predict category")
        print("2 - Display predictions")
        print("3 - Exit")
        choice = input("Enter your choice: ")

        if choice == '1':
            predict_category(classifier, vectorizer)
        elif choice == '2':
            display_some_predictions(y_pred, y_test, x_test)
        elif choice == '3':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please select a valid option.")
