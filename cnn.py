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
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tabulate import tabulate
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


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


def predict_category(classifier, tokenizer, label_encoder):
    title = input("Enter the title: ")
    abstract = input("Enter the abstract: ")

    combined_text = title + ' ' + abstract
    preprocessed_input = preprocess_text(combined_text)

    input_seq = tokenizer.texts_to_sequences([preprocessed_input])
    input_padded = pad_sequences(input_seq, maxlen=100, padding='post', truncating='post')

    predicted_label_index = classifier.predict(input_padded).argmax(axis=-1)
    predicted_label = label_encoder.inverse_transform(predicted_label_index)[0]

    print(f"Predicted category: {predicted_label}")


def display_some_predictions(predictions, actuals, texts, label_encoder):
    num_predictions_to_display = 20
    random_indices = random.sample(range(len(predictions)), num_predictions_to_display)
    display_data = []

    for i in random_indices:
        predicted_label = label_encoder.inverse_transform([predictions[i]])[0]
        actual_label = actuals.iloc[i]
        text = texts.iloc[i]
        display_data.append([predicted_label, actual_label, text])

    headers = ["Predicted", "Actual", "Text"]
    print("\nSome Predictions:")
    print(tabulate(display_data, headers=headers, tablefmt="grid"))


def plot_confusion_matrix(y_test, y_pred, label_encoder):
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.show()


def plot_precision_recall_curves(y_test, y_pred_prob, label_encoder):
    precision = dict()
    recall = dict()
    average_precision = dict()

    categories = label_encoder.classes_

    for i, category in enumerate(categories):
        category_idx = label_encoder.transform([category])[0]  # Convert category to index
        precision[category], recall[category], _ = precision_recall_curve(
            y_test == category_idx, y_pred_prob[:, category_idx])
        average_precision[category] = auc(recall[category], precision[category])

    plt.figure(figsize=(10, 8))

    for category in categories:
        plt.plot(recall[category], precision[category], lw=2,
                 label=f'{category} (AP = {average_precision[category]:.2f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')

    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))  # Adjust the bbox_to_anchor parameter

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid(True)
    plt.tight_layout()

    plt.show()

def run_cnn():
    dataset = pd.read_csv('articles.csv', delimiter='|')

    dataset.dropna(subset=['title', 'abstract', 'category'], inplace=True)

    dataset['text'] = dataset['title'] + ' ' + dataset['abstract']

    dataset['preprocessed_text'] = dataset['text'].apply(preprocess_text)

    x_train, x_test, y_train, y_test = train_test_split(dataset['preprocessed_text'], dataset['category'],
                                                        stratify=dataset['category'], test_size=0.2, random_state=42)

    tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
    tokenizer.fit_on_texts(x_train)
    x_train_seq = tokenizer.texts_to_sequences(x_train)
    x_test_seq = tokenizer.texts_to_sequences(x_test)
    x_train_padded = pad_sequences(x_train_seq, maxlen=100, padding='post', truncating='post')
    x_test_padded = pad_sequences(x_test_seq, maxlen=100, padding='post', truncating='post')

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    oversampler = RandomOverSampler(random_state=42)
    x_train_padded_resampled, y_train_resampled_encoded = oversampler.fit_resample(x_train_padded, y_train_encoded)

    cnn_model = Sequential([
        Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128, input_length=100),
        Conv1D(128, 5, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(64, activation='relu'),
        Dense(len(label_encoder.classes_), activation='softmax')
    ])

    cnn_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    cnn_model.fit(x_train_padded_resampled, y_train_resampled_encoded, validation_data=(x_test_padded, y_test_encoded),
                  epochs=5, batch_size=32)

    y_pred_prob = cnn_model.predict(x_test_padded)
    y_pred = y_pred_prob.argmax(axis=-1)

    categories = label_encoder.classes_

    class_report = classification_report(y_test_encoded, y_pred, target_names=categories, zero_division='warn')
    print(class_report)

    plot_confusion_matrix(y_test_encoded, y_pred, label_encoder)
    plot_precision_recall_curves(y_test_encoded, y_pred_prob, label_encoder)

    while True:
        print("\nMenu:")
        print("1 - Predict category")
        print("2 - Display predictions")
        print("3 - Exit")
        choice = input("Enter your choice: ")

        if choice == '1':
            predict_category(cnn_model, tokenizer, label_encoder)
        elif choice == '2':
            display_some_predictions(y_pred, y_test, x_test, label_encoder)
        elif choice == '3':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please select a valid option.")
