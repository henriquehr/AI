
import sys
import os.path
import time
import zipfile
import gensim
import numpy as np
import sklearn as sk
from sklearn import neighbors
import matplotlib.pyplot as plt


def load_base(zip_name):
    if shuffle_database:
        print("Loading and shuffling the database...")
    else:
        print("Loading the database...")
    with zipfile.ZipFile(zip_name) as zip:
        file_name = ""
        for files in zip.namelist():
            if '.csv' in files and '/' not in files:
                file_name = files
        if file_name is "":
            print("Must be the .zip file with a .csv inside.")
            quit()
        with zip.open(file_name, 'r') as file:
            lines =  np.asarray(file.readlines()[1:])
            if shuffle_database:
                shuf = np.arange(len(lines))
                np.random.shuffle(shuf)
                lines = lines[shuf]
            data_w2v = np.empty(50000, dtype=np.object)
            data_test = np.empty(10000, dtype=np.object)
            labels_test = np.empty(10000, dtype=np.object)
            data_train = np.empty(40000, dtype=np.object)
            labels_train = np.empty(40000, dtype=np.object)
            d_tr_i = 0
            d_te_i = 0
            d_wv_i = 0
            for i in range(len(lines)):
                s = str(lines[i]).split(',')
                l = len(s)
                rev = ''.join(s[2:l-2])
                feel = ''.join(s[l-2:l-1])
                g = gensim.utils.simple_preprocess(rev)
                if feel == 'pos' or feel == 'neg':
                    if d_te_i < 10000:
                        data_test[d_te_i] = g
                        labels_test[d_te_i] = feel
                        d_te_i += 1
                    elif d_tr_i < 50000:
                        data_train[d_tr_i] = g
                        labels_train[d_tr_i] = feel
                        d_tr_i += 1
                    else:
                        data_w2v[d_wv_i] = g
                        d_wv_i += 1
                else:
                    data_w2v[d_wv_i] = g
                    d_wv_i += 1
            return data_w2v, data_train, labels_train, data_test, labels_test


def words_mean(model, sentences):
    return [(each_word_mean(model, s)) for s in sentences]


def each_word_mean(model, sentence):
    return [np.mean(model.wv[word]) for word in sentence if word in model.wv.vocab]


def make_same_length(data_train, data_test):
    print("Making all the vectors the same length...")
    m = np.max([len(_) for _ in data_test])
    m1 = np.max([len(_) for _ in data_train])
    big = np.max((m,m1))
    v1 = [(np.pad(_, (0, abs(len(_) - big)), 'constant')) for _ in data_train]
    v2 = [(np.pad(_, (0, abs(len(_) - big)), 'constant')) for _ in data_test]
    return v1, v2


def knn_table(data_train, labels_train):
    print("Making the knn classifier table...")
    return [((data_train[i], labels_train[i])) for i in range(len(data_train))]


def metric_euclidean_distance(v1, v2):
    return np.sqrt(np.sum(np.power((v1 - v2), 2)))


def metric_mean_distance(v1, v2):
    _v1 = np.mean(v1)
    _v2 = np.mean(v2)
    return np.sqrt(np.power((_v1 - _v2), 2))


def metric_only_sum(v1, v2):
    return abs(np.sum(v1) - np.sum(v2))


def metric_vector_length(v1, v2):
    l1 = np.sqrt(np.sum(np.power(v1, 2)))
    l2 = np.sqrt(np.sum(np.power(v2, 2)))
    return abs(l1 - l2)


def knn_classify(k, knn_tab, data_test, labels_test):
    print("Classifying...")
    print("It will take a loooooooooooooooooooooooong time if using too many sentences...")
    corrects = 0
    order_again = True
    labels_predicted = []
    for data_idx in range(len(data_test)):
        testing = data_test[data_idx]
        closer_k = knn_tab[:k]
        nearest_lenght = metric(closer_k[0][0], testing)
        knn_idx_out = k
        for _ in range(k, len(knn_tab)):
            if order_again:
                for _n in range(len(closer_k)):
                    b = metric(closer_k[_n][0], testing)
                    if nearest_lenght < b:
                        tmp = closer_k[0]
                        closer_k[0] = closer_k[_n]
                        closer_k[_n] = tmp
                        nearest_lenght = b
                        order_again = False
            next = metric(knn_tab[knn_idx_out][0], testing)
            if nearest_lenght > next:
                nearest_lenght = next
                closer_k[0] = knn_tab[knn_idx_out]
                order_again = True
            knn_idx_out += 1
        pos_neg = 0
        for n in closer_k:
            if n[1] == 'neg':
                pos_neg -= 1
            else:
                pos_neg += 1
        if pos_neg > 0:
            labels_predicted.append('pos')
            if labels_test[data_idx] == 'pos':
                corrects += 1
        elif pos_neg < 0:
            labels_predicted.append('neg')
            if labels_test[data_idx] == 'neg':
                corrects += 1
    return (corrects / len(data_test)), labels_predicted


def knn_scikit(k, data_train, labels_train, data_test, labels_test):
    print("Classifying with scikit knn...")
    jobs = 4
    print("K = " + str(k))
    print("Metric = " + str(metric))
    print("Parallel jobs = " + str(jobs))
    clf = sk.neighbors.KNeighborsClassifier(n_neighbors=k, n_jobs=jobs, metric=metric)
    print("\"Training\"...")
    clf.fit(data_train, labels_train)
    print("Computing predictions...")
    labels_predicted = clf.predict(data_test)
    print("Computing accuracy...")
    score = np.mean(labels_predicted ==  labels_test) # faster than computing the score
    #score = clf.score(data_test, labels_test)
    return score, labels_predicted


def show_confusion_matrix(labels_test, labels_predicted):
    cm = sk.metrics.confusion_matrix(labels_test, labels_predicted)
    return cm


use_scikit_knn = True # Use the knn implemented or the scikit knn
reuse_w2v_model = False # If there's a w2v model saved, use it
shuffle_database = True # Shuffle or not the order of the sentences in the database

if use_scikit_knn:
    metric = 'euclidean' # manhattan euclidean jaccard chebyshev
else:
    metric = metric_euclidean_distance # metric_only_sum metric_vector_length metric_mean_distance

if __name__ == "__main__":

    if len(sys.argv) != 3:
        sys.exit("Use: python " + str(__file__) + " zip_name k_value")

    start = time.time()

    zip_name = sys.argv[1]
    k = int(sys.argv[2])

    model_file_name = "word2vec.model"

    data_w2v, data_train, labels_train, data_test, labels_test = load_base(zip_name)

    print("Word2Vec: " + str(len(data_w2v)) + " sentences.")
    print("Train: " + str(len(data_train)) + " sentences.")
    print("Test: " + str(len(data_test)) + " sentences.")

    if os.path.isfile(model_file_name) and reuse_w2v_model:
        print("Saved w2v model found. Loading w2v model...")
        model_w2v = gensim.models.Word2Vec.load(model_file_name)
    else:
        print("Creating w2v model...")
        model_w2v = gensim.models.Word2Vec(data_w2v, size=100, window=20, min_count=1, workers=10)
        print("Training w2v model...")
        model_w2v.train(data_w2v, total_examples=len(data_w2v), epochs=10)
        print("Saving w2v model...")
        model_w2v.save(model_file_name)

    del data_w2v
    
    print("Computing the mean of each word vector...")
    data_train = words_mean(model_w2v, data_train)
    data_test = words_mean(model_w2v, data_test)

    data_train, data_test = make_same_length(data_train, data_test)

    if not use_scikit_knn:
        knn_tab = knn_table(data_train, labels_train)
        score, labels_predicted = knn_classify(k, knn_tab, data_test, labels_test)
    else:
        score, labels_predicted = knn_scikit(k, data_train, labels_train, data_test, labels_test)
    
    cm = show_confusion_matrix(labels_test, labels_predicted)

    print("Confusion matrix: ")
    print(cm)
    print("Accuracy: " + str(score * 100) + " %")

    end = time.time()
    print("Took: " + str(round((end - start), 3)) + " seconds to finish.")
    plt.matshow(cm)
    plt.colorbar()
    plt.show()
