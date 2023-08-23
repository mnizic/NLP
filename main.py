import csv

from cnn import run_cnn
from logisticRegression import run_logistic_regression_with_ngrams
from naiveBayes import run_naive_bayes_with_ngrams
import time


def extract_category(category_string):
    category = category_string.split(';')[0].strip()
    category = category.rstrip('"')
    category = category.replace('"', '').strip()

    return category


def remove_rows_with_low_category_count(csv_file, category_column_index, min_category_count):
    category_counts = {}
    rows_to_keep = []

    with open(csv_file, 'r', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile, delimiter='|')
        header = next(csvreader)
        rows_to_keep.append(header)

        for row in csvreader:
            category = extract_category(row[category_column_index])
            if category not in category_counts:
                category_counts[category] = 1
            else:
                category_counts[category] += 1

    with open(csv_file, 'r', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile, delimiter='|')
        header = next(csvreader)

        for row in csvreader:
            category = extract_category(row[category_column_index])
            if category_counts[category] >= min_category_count:
                rows_to_keep.append(row)

    with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter='|', quoting=csv.QUOTE_NONE, escapechar='\\')
        csvwriter.writerows(rows_to_keep)


def txt_to_csv(txt_file, csv_file):
    with open(txt_file, 'r', encoding='utf-8') as txt_f, open(csv_file, 'w', newline='', encoding='utf-8') as csv_f:
        csv_writer = csv.writer(csv_f, delimiter='|', quoting=csv.QUOTE_NONE, escapechar='\\')
        csv_writer.writerow(['title', 'abstract', 'category'])  # Write header row

        title = ""
        abstract = ""
        category = ""

        for line in txt_f:
            line = line.strip()
            if line.startswith("TI "):
                title = line[3:]
            elif line.startswith("AB "):
                extracted_category = extract_category(category)
                title = title.replace('"', '').strip()
                abstract = abstract.replace('"', '').strip()
                extracted_category = extracted_category.replace('"', '').strip()
                csv_writer.writerow([title, abstract, extracted_category])
                abstract = line[3:]  # Start new abstract
            elif line.startswith("WC "):
                category = line[3:]
            else:
                if title:
                    title += " " + line
                if abstract:
                    abstract += " " + line

        if title and abstract and category:
            extracted_category = extract_category(category)
            csv_writer.writerow([title, abstract, extracted_category])


def remove_duplicates_from_csv(csv_filename, column_index):
    seen_values = set()
    rows_to_keep = []

    with open(csv_filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        header = next(csvreader)
        rows_to_keep.append(header)

        for row in csvreader:
            value = row[column_index]
            if value not in seen_values:
                seen_values.add(value)
                rows_to_keep.append(row)

    with open(csv_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(rows_to_keep)


def remove_unnecessary_categories(csv_data):
    with open(csv_data, 'r', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile, delimiter='|')
        header = next(csvreader)
        rows_to_keep = [header]

        for row in csvreader:
            categories = row[category_column_index]
            if any(keyword in categories for keyword in ['Psychology, Applied',
                                                         'Green & Sustainable Science & Technology',
                                                         'Psychology, Experimental',
                                                         'Behavioral Sciences',
                                                         'Computer Science',
                                                         'Engineering']):
                continue
            rows_to_keep.append(row)

    with open(csv_data, 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter='|', quoting=csv.QUOTE_NONE, escapechar='\\')
        csvwriter.writerows(rows_to_keep)


if __name__ == '__main__':
    start_time = time.time()
    print(time.time())
    input_txt_file = 'articles.txt'
    dataset = 'articles.csv'

    column_index_to_check = 0
    min_category_count = 300
    category_column_index = 2

    txt_to_csv(input_txt_file, dataset)
    remove_duplicates_from_csv(dataset, column_index_to_check)
    remove_rows_with_low_category_count(dataset, category_column_index, min_category_count)
    remove_unnecessary_categories(dataset)

    ngram_ranges_to_try = [(1, 1), (1, 2), (1, 3)]
    algorithm_choice = input("Choose algorithm (1 - Naive Bayes, 2 - Logistic Regression, 3 - CNN): ")

    if algorithm_choice == "1":
        print("Running Naive Bayes...")
        for ngram_range in ngram_ranges_to_try:
            print(f"Trying n-gram range: {ngram_range}")
            run_naive_bayes_with_ngrams(ngram_range)
            print(time.time() - start_time)
    elif algorithm_choice == "2":
        print("Running Logistic Regression...")
        for ngram_range in ngram_ranges_to_try:
            print(f"Trying n-gram range: {ngram_range}")
            run_logistic_regression_with_ngrams(ngram_range)
            print(time.time() - start_time)
    elif algorithm_choice == "3":
        print("Running Convolutional Neural Networks...")
        run_cnn()
        print(time.time() - start_time)
    else:
        print("Invalid algorithm choice")
