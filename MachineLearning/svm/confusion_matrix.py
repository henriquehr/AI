
import sys
from sklearn import metrics
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

if __name__ == "__main__":
    args_l = len(sys.argv)
    if ((args_l - 1) % 2) != 0 or args_l < 3:
        print("Arguments not matching. Must be pairs of predicted and tested files")
        print("Use: python " + __file__ + " predict test ...")
        print("predict = output file with predictions from svm")
        print("test = test file used to test the svm")
        print("... = optional, other pairs of predict and test")
        quit()
    else:
        for idx in range(1, args_l, 2):
            _predicted = sys.argv[idx]
            _tested = sys.argv[idx+1]
            with open(_predicted) as file:
                lines = file.readlines()
                if 'labels' in lines[0]:
                    lines = lines[1:]
                s_predicted = [l[0] for l in lines]
            with open(_tested) as file:
                lines = file.readlines()
                s_tested = [l[0] for l in lines]

            print(classification_report(s_tested, s_predicted))
            m_s = metrics.confusion_matrix(s_tested, s_predicted)
            print("Tested: " + str(len(s_tested)))
            print(m_s)
            plt.matshow(m_s)
            plt.colorbar()
        plt.show()
