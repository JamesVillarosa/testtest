import tkinter as tk                                                        # Importing for GUI
import os                                                                   # For writing and reading file
import re                                                                   # For regular expression
import random                                                               # For shuffling data
import matplotlib.pyplot as plt                                             # For plotting the matrics
from tkinter import filedialog
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix     # For evaluating model performance


def read_folder(folder_path):
    vocabulary = []                                                         # Array for all words
    words_frequency = {}                                                    # Dictionary for unique words and their frequency

    dir_list = os.listdir(folder_path)                                      # Getting all file inside the folder
    for file_name in dir_list:                                              # Traversing all file inside folder
        file_path = os.path.join(folder_path, file_name)                    # Constructing the file path               

        if os.path.isfile(file_path):                                       # Check if file path exist
            with open(file_path, 'r', encoding='latin-1') as f:             # Read the file in latin-1
                tokenize = f.read().split()                                 # Read the content of the file and splitting the words by white spaces
                regex = re.compile('[^a-zA-Z]')                             # Initializing regex for small and big alphabet only

                for i in tokenize:                                          # We will clean the tokenize words
                    clean = regex.sub('', i).lower()                        # Removing the non alphabet
                    if clean:
                        vocabulary.append(clean)                            # If word exist, append in vocabulary


    unique_words = set(vocabulary)                                          # Get rid of the duplicates
    for word in unique_words:
        words_frequency[word] = vocabulary.count(word)                      # Traversing the unique words and appending the words in dictionary together with its frequency

    return vocabulary, words_frequency                                      # Return the vocubularu and the words and their frequency



def split_data(data, split_ratio=(0.6, 0.2, 0.2)):
    random.shuffle(data)                                                    # Shuffle the input data
    train_size = int(split_ratio[0] * len(data))                            # Calculate the size for training data
    val_size = int(split_ratio[1] * len(data))                              # Calculate the size for validation data

    train_data = data[:train_size]                                          # Split training data
    val_data = data[train_size:train_size + val_size]                       # Split validation data
    test_data = data[train_size + val_size:]                                # Remaining data is test data

    return train_data, val_data, test_data                                  # Return the split datasets



def calculate_probabilities(bow_spam, bow_ham, classify_files, smoothing_factor):
    spam_total_words = len(bow_spam[0])                                     # Total words in spam
    ham_total_words = len(bow_ham[0])                                       # Total words in ham    
    spam_unique_words = len(bow_spam[1])                                    # Unique words in spam
    ham_unique_words = len(bow_ham[1])                                      # Unique words in ham

    spam_word_probabilities = {}                                            # Dictionaries to store word probabilities
    ham_word_probabilities = {}

    for word in bow_spam[1]:                                                # Calculate probabilities for words in spam
        word_count = bow_spam[1].get(word, 0)                               # Get count of the word in spam
        spam_word_probabilities[word] = (word_count + smoothing_factor) / (spam_total_words + smoothing_factor * spam_unique_words)

    for word in bow_ham[1]:                                                 # Calculate probabilities for words in ham
        word_count = bow_ham[1].get(word, 0)                                # Get count of the word in ham
        ham_word_probabilities[word] = (word_count + smoothing_factor) / (ham_total_words + smoothing_factor * ham_unique_words)

    classifications = []                                                    # List to store the classification results

    for file_name, clean_words in classify_files:                           # Classify each file in the classification folder
        spam_prob = 1                                                       # Initialize probabilities for spam and ham
        ham_prob = 1

        for word in clean_words:                                            # Calculate the probabilities for the words in the cleaned content
            spam_prob *= spam_word_probabilities.get(word, smoothing_factor / (spam_total_words + smoothing_factor * spam_unique_words))
            ham_prob *= ham_word_probabilities.get(word, smoothing_factor / (ham_total_words + smoothing_factor * ham_unique_words))

        if spam_prob > ham_prob:                                            # Determine if the file is classified as Spam or Ham
            classifications.append((file_name, "Spam"))
        else:
            classifications.append((file_name, "Ham"))

    return classifications                                                  # Return the classification results


def evaluate_model(classifications, true_labels):
    y_pred = [label for _, label in classifications]                        # Extract predicted labels from classifications
    accuracy = accuracy_score(true_labels, y_pred)                          # Calculate accuracy of the predictions
    precision = precision_score(true_labels, y_pred, pos_label="Spam")      # Calculate precision
    recall = recall_score(true_labels, y_pred, pos_label="Spam")            # Calculate recall
    conf_matrix = confusion_matrix(true_labels, y_pred, labels=["Spam", "Ham"])  # Generate confusion matrix

    TP = conf_matrix[0, 0]                                                  # True positives
    TN = conf_matrix[1, 1]                                                  # True negatives
    FP = conf_matrix[1, 0]                                                  # False positives
    FN = conf_matrix[0, 1]                                                  # False negatives

    return accuracy, precision, recall, conf_matrix, TP, TN, FP, FN         # Return evaluation metrics


def plot_metrics(k_values, accuracies, precisions, recalls):
    plt.plot(k_values, accuracies, label="Accuracy")                        # Plot accuracy
    plt.plot(k_values, precisions, label="Precision")                       # Plot precision
    plt.plot(k_values, recalls, label="Recall")                             # Plot recall
    plt.xlabel("Laplace Smoothing Factor (k)")                              # Label for x-axis
    plt.ylabel("Score")                                                     # Label for y-axis
    plt.legend()                                                            # Show legend
    plt.title("Model Performance vs. Smoothing Factor (k)")                 # Title of the plot
    plt.show()                                                              # Display the plot


def classify_and_evaluate(folder_path):
    spam_folder = os.path.join(folder_path, 'spam')                         # Path to spam folder
    ham_folder = os.path.join(folder_path, 'ham')                           # Path to ham folder

    spam_files = [(file_name, read_clean_words(os.path.join(spam_folder, file_name))) for file_name in os.listdir(spam_folder)]     # Read and clean words from spam files
    ham_files = [(file_name, read_clean_words(os.path.join(ham_folder, file_name))) for file_name in os.listdir(ham_folder)]        # Read and clean words from ham files

    train_spam, val_spam, test_spam = split_data(spam_files)                # Split spam files into training, validation, and test sets
    train_ham, val_ham, test_ham = split_data(ham_files)                    # Split ham files into training, validation, and test sets

    train_files = train_spam + train_ham                                    # Combine training sets
    val_files = val_spam + val_ham                                          # Combine validation sets
    test_files = test_spam + test_ham                                       # Combine test sets

    true_val_labels = ["Spam"] * len(val_spam) + ["Ham"] * len(val_ham)     # True labels for validation set
    true_test_labels = ["Spam"] * len(test_spam) + ["Ham"] * len(test_ham)  # True labels for test set

    k_values = [0.005, 0.01, 0.5, 1.0, 2.0]                                 # Different smoothing factors to evaluate
    accuracies, precisions, recalls = [], [], []                            # Lists to store evaluation metrics

    bow_spam = read_folder(spam_folder)                                     # Read spam words and their frequency
    bow_ham = read_folder(ham_folder)                                       # Read ham words and their frequency

    for k in k_values:                                                      # Iterate over each smoothing factor
        classifications = calculate_probabilities(bow_spam, bow_ham, val_files, k)                      # Classify validation files
        accuracy, precision, recall, _, _, _, _, _ = evaluate_model(classifications, true_val_labels)   # Evaluate model performance
        accuracies.append(accuracy)                                         # Store accuracy
        precisions.append(precision)                                        # Store precision
        recalls.append(recall)                                              # Store recall

    plot_metrics(k_values, accuracies, precisions, recalls)                 # Plot the metrics

    best_k = k_values[accuracies.index(max(accuracies))]                    # Find the best k value based on accuracy
    print(f"Best k value: {best_k}")                                        # Print the best k value

    final_classifications = calculate_probabilities(bow_spam, bow_ham, test_files, best_k)  # Classify test files using best k
    accuracy, precision, recall, conf_matrix, TP, TN, FP, FN = evaluate_model(final_classifications, true_test_labels)  # Evaluate test performance

    print(f"Test Accuracy: {accuracy}")                                     # Print test accuracy
    print(f"Test Precision: {precision}")                                   # Print test precision
    print(f"Test Recall: {recall}")                                         # Print test recall
    print("Confusion Matrix:")                                              # Print confusion matrix
    print(f"[[TP: {TP}, FN: {FN}]")                                         # Display true positives and false negatives
    print(f" [FP: {FP}, TN: {TN}]]")                                        # Display false positives and true negatives


def read_clean_words(file_path):
    with open(file_path, 'r', encoding='latin-1') as f:                     # Open file to read content
        content = f.read().split()                                          # Read content and split into words
        regex = re.compile('[^a-zA-Z]')                                     # Regex to filter out non-alphabetic characters
        clean_words = [regex.sub('', token).lower() for token in content if regex.sub('', token)]   # Cleaning the words
    return clean_words                                                      # Return cleaned words


def select_folder():
    root = tk.Tk()                                                          # Initialize the Tkinter root window
    root.withdraw()                                                         # Hide the main window

    folder_path = filedialog.askdirectory()                                 # Getting directory from user
    classify_and_evaluate(folder_path)                                      # Start classification and evaluation


select_folder()                                                             # Call the folder selection function
