import sys
import time
import zipfile
import numpy as np
import matplotlib.pyplot as plt
from sklearn import base
from sklearn import ensemble
from sklearn.datasets import load_svmlight_file
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

def usage(valid_args):
    print("Usage: ")
    print("python " + str(__file__) + " .zip")
    print("python " + str(__file__) + " .zip " + valid_args[0])
    print("python " + str(__file__) + " .zip " + valid_args[1])
    print("python " + str(__file__) + " .zip " + valid_args[2])
    print("python " + str(__file__) + " .zip " + valid_args[3])
    print("python " + str(__file__) + " .zip " + valid_args[4])
    print("python " + str(__file__) + " .zip " + valid_args[5])
    print("python " + str(__file__) + " .zip " + valid_args[6])
    print("python " + str(__file__) + " .zip " + valid_args[7])
    print("python " + str(__file__) + " .zip " + valid_args[8])
    print("python " + str(__file__) + " .zip " + valid_args[9])
    

def load_database(path):
    print("Loading .zip file...")
    with zipfile.ZipFile(path) as zip:
        files = [f for f in zip.namelist() if "txt" in f and "__MAC" not in f]
        with zip.open(files[0] if "train" in files[0] else files[1]) as file: 
            data_train = load_svmlight_file(file)
        with zip.open(files[1] if "test" in files[1] else files[0]) as file:
            data_test = load_svmlight_file(file)
        print("Train samples: " + str(np.shape(data_train[0])))
        print("Test samples: " + str(np.shape(data_test[0])))
    return data_train[0], data_train[1], data_test[0], data_test[1]


def clf_train_print_score(clf, data_train, labels_train, data_test, labels_test):
    print("     Trainning classifier...")
    _s = time.time()
    clf.fit(data_train, labels_train)
    print("     Computing score...")
    score = clf.score(data_test, labels_test)
    t = time.time() - _s
    print("     Accuracy: " + str(round(score * 100, 4)) + "%  Time : "  + str(t) + "s")
    print("\n")
    return score * 100, t


def plot_graph(scores, times, x_from, x_max, x_jump, title, xlabel):
    print("Plotting graph...")
    xtick = np.arange(x_from, x_max + x_jump, x_jump)    
    _, ax1 = plt.subplots()
    plt.title(title)
    a1, = ax1.plot(xtick, scores, label="Accuracy", color="b")
    ax1.set_ylabel("Accuracy (%)", color="b")
    ax1.tick_params("y", colors="b")
    ax1.set_xlabel(xlabel)
    ax2 = ax1.twinx()
    a2, = ax2.plot(xtick, times, label="Times", color="r")
    ax2.set_ylabel("Time (s)", color="r")
    ax2.tick_params("y", colors="r")
    plt.legend([a1, a2], ["Accuracy (%)", "Time (s)"])
    plt.xticks(xtick)
    plt.show()

def plot_graph_vs(scores, times, x_from, x_max, x_jump, title, xlabel):
    print("Plotting graph...")
    xtick = np.arange(x_from, x_max + x_jump, x_jump)    
    _, ax1 = plt.subplots()
    plt.title(title)
    label11 = scores[0][0]
    vec = scores[0][1]
    color = "b"
    a11, = ax1.plot(xtick, vec, label=label11, color=color)
    label12 = scores[1][0]
    vec = scores[1][1]
    a12, = ax1.plot(xtick, vec, label=label12, color=color, linestyle="--")
    ax1.set_ylabel("Accuracy (%)", color=color)
    ax1.tick_params("y", colors=color)
    ax1.set_xlabel(xlabel)
    ax2 = ax1.twinx()
    label21 = times[0][0]
    vec = times[0][1]
    color = "r"
    a21, = ax2.plot(xtick, vec, label=label21, color=color)
    label22 = times[1][0]
    vec = times[1][1]
    a22, = ax2.plot(xtick, vec, label=label22, color=color, linestyle="--")
    ax2.set_ylabel("Time (s)", color=color)
    ax2.tick_params("y", colors=color)
    plt.legend([a11, a21, a12, a22], [str(label11) + " Accuracy (%)", str(label21) + " Time (s)", str(label12) + " Accuracy (%)", str(label22) + " Time (s)"])
    plt.xticks(xtick)
    plt.show()

def run_test_estimators(clf, estimators, est_start, est_jump, title, data_train, labels_train, data_test, labels_test, plot):
    scores = []
    times = []
    print("\n")
    print(">>>> " + title + " <<<<")
    print("\n")
    clf_copy = base.clone(clf)
    for e in range(est_start, estimators + (est_jump), est_jump):
        print(" Estimators : " + str(e))
        clf_copy.set_params(n_estimators=e)
        score, t = clf_train_print_score(clf_copy, data_train, labels_train, data_test, labels_test)
        scores.append(score)
        times.append(t)
        clf_copy = base.clone(clf)
    if plot:
        plot_graph(scores, times, est_start, estimators, est_jump, title, "Estimators")
    else:
        return scores, times

def run_test_lr(clf, lr_tests, growth, title, data_train, labels_train, data_test, labels_test):
    scores = []
    times = []
    print("\n")
    print(">>>> " + title + " <<<<")
    print("\n")
    clf_copy = base.clone(clf)
    lr = 0.1
    for _ in range(0, lr_tests):
        print(" Learning rate: " + str(lr))
        clf_copy.set_params(learning_rate=lr)
        score, t = clf_train_print_score(clf_copy, data_train, labels_train, data_test, labels_test)
        scores.append(score)
        times.append(t)
        clf_copy = base.clone(clf)
        lr += growth
    plot_graph(scores, times, 0.1, 1.0, growth, title, "Learning Rate")

if __name__ == "__main__":
    classifier = "all"
    valid_args = ["all", "bagging_dt", "bagging_mlp", "a_boost", "g_boosting", "rf", "e_tree", "bagging", "boost", "trees"]
    njobs = 4
    bagging = {'estimators': 30, 'start': 2, 'jump': 4}
    boost = {'estimators': 150, 'start': 10, 'jump': 10, 'lr_tests': 10, 'lr_growth': 0.1}
    trees = {'estimators': 30, 'start': 2, 'jump': 4}
    
    if len(sys.argv) == 2:
        path = sys.argv[1]
        data_train, labels_train, data_test, labels_test = load_database(path)
    elif len(sys.argv) == 3:
        classifier = sys.argv[2]
        if classifier in valid_args:
            path = sys.argv[1]
            data_train, labels_train, data_test, labels_test = load_database(path)
        else:
            usage(valid_args)
            sys.exit()
    else:
        usage(valid_args)
        sys.exit()

    print("")
    print("Testing: " + str(classifier))


    if classifier == valid_args[1] or classifier == valid_args[0]:
        title = "Bagging + Decision Tree"
        clf = ensemble.BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_jobs=njobs)
        run_test_estimators(clf, bagging['estimators'], bagging['start'], bagging['jump'], title, data_train, labels_train, data_test, labels_test, True)
    if classifier == valid_args[2] or classifier == valid_args[0]:
        title = "Bagging + MLP"
        clf = ensemble.BaggingClassifier(base_estimator=MLPClassifier(), n_jobs=njobs)
        run_test_estimators(clf, bagging['estimators'], bagging['start'], bagging['jump'], title, data_train, labels_train, data_test, labels_test, True)
    if classifier == valid_args[3] or classifier == valid_args[0]:
        title = "Ada Boost"
        clf = ensemble.AdaBoostClassifier()
        run_test_estimators(clf, boost['estimators'], boost['start'], boost['jump'], title, data_train, labels_train, data_test, labels_test, True)
        run_test_lr(clf, boost['lr_tests'], boost['lr_growth'], title, data_train, labels_train, data_test, labels_test)
    if classifier == valid_args[4] or classifier == valid_args[0]:
        title = "Gradient Boosting"
        clf = ensemble.GradientBoostingClassifier()
        run_test_estimators(clf, boost['estimators'], boost['start'], boost['jump'], title, data_train, labels_train, data_test, labels_test, True)
        run_test_lr(clf, boost['lr_tests'], boost['lr_growth'], title, data_train, labels_train, data_test, labels_test)
    if classifier == valid_args[5] or classifier == "all":
        title = "Random Forest"
        clf = ensemble.RandomForestClassifier(n_jobs=njobs)
        run_test_estimators(clf, trees['estimators'], trees['start'], trees['jump'], title, data_train, labels_train, data_test, labels_test, True)
    if classifier == valid_args[6] or classifier == valid_args[0]:
        title = "Extra Trees"
        clf = ensemble.ExtraTreesClassifier(n_jobs=njobs)
        run_test_estimators(clf, trees['estimators'], trees['start'], trees['jump'], title, data_train, labels_train, data_test, labels_test, True)
   
    if classifier == valid_args[7]:
        title = "Bagging + Decision Tree"
        clf = ensemble.BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_jobs=njobs)
        scores, times = run_test_estimators(clf, bagging['estimators'], bagging['start'], bagging['jump'], title, data_train, labels_train, data_test, labels_test, False)
        title = "Bagging + MLP"
        clf2 = ensemble.BaggingClassifier(base_estimator=MLPClassifier(), n_jobs=njobs)
        scores2, times2 = run_test_estimators(clf2, bagging['estimators'], bagging['start'], bagging['jump'], title, data_train, labels_train, data_test, labels_test, False)
        title = "Bagging + Decision Tree vs Bagging + MLP"
        _scores = (("Bagging + Decision Tree", scores), ("Bagging + MLP", scores2))
        _times = (("Bagging + Decision Tree", times), ("Bagging + MLP", times2))
        plot_graph_vs(_scores, _times, bagging['start'], bagging['estimators'], bagging['jump'], title, "Estimators")
    if classifier == valid_args[8]:
        title = "Ada Boost"
        clf = ensemble.AdaBoostClassifier()
        scores, times = run_test_estimators(clf, boost['estimators'], boost['start'], boost['jump'], title, data_train, labels_train, data_test, labels_test, False)
        title = "Gradient Boosting"
        clf2 = ensemble.GradientBoostingClassifier()
        scores2, times2 = run_test_estimators(clf2, boost['estimators'], boost['start'], boost['jump'], title, data_train, labels_train, data_test, labels_test, False)
        title = "Ada Boost vs Gradient Boosting"
        _scores = (("Ada Boost", scores), ("Gradient Boosting", scores2))
        _times = (("Ada Boost", times), ("Gradient Boosting", times2))
        plot_graph_vs(_scores, _times, boost['start'], boost['estimators'], boost['jump'], title, "Estimators")
    if classifier == valid_args[9]:
        title = "Extra Trees"
        clf = ensemble.ExtraTreesClassifier(n_jobs=njobs)
        scores, times = run_test_estimators(clf, trees['estimators'], trees['start'], trees['jump'], title, data_train, labels_train, data_test, labels_test, False)
        title = "Random Forest"
        clf2 = ensemble.RandomForestClassifier(n_jobs=njobs)
        scores2, times2 = run_test_estimators(clf2, trees['estimators'], trees['start'], trees['jump'], title, data_train, labels_train, data_test, labels_test, False)
        title = "Extra Trees vs Random Forest"
        _scores = (("Random Forest", scores), ("Extra Trees", scores2))
        _times = (("Random Forest", times), ("Extra Trees", times2))
        plot_graph_vs(_scores, _times, trees['start'], trees['estimators'], trees['jump'], title, "Estimators")
