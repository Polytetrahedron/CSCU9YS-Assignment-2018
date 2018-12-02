import os
from sklearn import svm, metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import DatasetProcessing as processing
import matplotlib.pyplot as plt
import random as ran


def find_directory(selection: int):
    """
    This function creates a path to a the directory where the dataset is stored

    :param selection: The dataset to navigate to
    :return dirpath: The path to the dataset
    """
    path = ""
    if selection == 1:
        path = "train-mails"
    elif selection == 2:
        path = "test-mails"
    elif selection > 2:
        path = "all-mails"
    dirpath = os.path.dirname(os.path.realpath(__file__))
    dirpath = os.path.join(dirpath, "spam-non-spam-dataset\\" + path)
    return dirpath


def svm_classifier():
    """
    This is the SVM classifier.
    :return:
    """
    print('')
    print("\nRunning SVM...")
    clf = svm.SVC(kernel='linear')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
    print("Precision:", metrics.recall_score(y_test, y_pred))
    print("Recall:", metrics.recall_score(y_test, y_pred))
    print("F1:", metrics.f1_score(y_test, y_pred))
    print("Confusion Matrix \n"
          "nonSpam  Spam\n")
    print(metrics.confusion_matrix(y_test, y_pred))


def mlp_classifier():
    """
    This is the Multilayer Perceptron classifier.
    It is running using all of the default values (e.g number of hidden layers)
    :return:
    """
    print("\nRunning MLP")
    mlp = MLPClassifier()
    mlp.fit(X_train, y_train)
    mlp_pred = mlp.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, mlp_pred))
    print("Precision:", metrics.recall_score(y_test, mlp_pred))
    print("Recall:", metrics.recall_score(y_test, mlp_pred))
    print("F1:", metrics.f1_score(y_test, mlp_pred))
    print("Confusion Matrix \n"
          "nonSpam  Spam \n")
    print(processed_dataset.target_labels, metrics.confusion_matrix(y_test, mlp_pred))


def nb_classifier():
    """
    This is the Naive Bayes Classifier
    :return:
    """
    print("\nRunning Naive Bayes")
    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    nb_pred = nb.predict(X_test)
    print("Accuracy: ", metrics.accuracy_score(y_test, nb_pred))
    print("Precision:", metrics.recall_score(y_test, nb_pred))
    print("Recall:", metrics.recall_score(y_test, nb_pred))
    print("F1:", metrics.f1_score(y_test, nb_pred))
    print("Confusion Matrix \n"
          "nonSpam  Spam \n")
    print(metrics.confusion_matrix(y_test, nb_pred))


"""
These lines will run the dataset preprocessing, the test train split function and then run all three classifiers

To select the directory for testing 1 = training mails, 2 = test mails, 3 = full dataset
I would recommend running the full dataset as it gives the most accurate representation of classifier performance.

The data in all three directories is split in a 50/50 (481 training/481 test on full set) split using the sklearn 
split function
"""
processed_dataset = processing.process_dataset(find_directory(3))
X_train, X_test, y_train, y_test = train_test_split(processed_dataset.data, processed_dataset.target, test_size=0.50,
                                                    random_state=109)

svm_classifier()
mlp_classifier()
nb_classifier()

