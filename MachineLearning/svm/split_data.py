
import sys


def save_file(file_name, content):
    location = file_name + ".scale"
    with open(location,'w') as file:
        file.writelines(content)
        print(location + " saved.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Use: python " + __file__ + " data.scale easy_size")
        print("data.scale = normalized database.")
        print("easy_size = amount of samples to use on easy.py, or all for all the samples (test[:easy_size], train[easy_size:easy_size+easy_size]).")
        quit()
    file_name = sys.argv[1]
    easy_size = sys.argv[2]
        
    with open(file_name) as file:
        lines = file.readlines()
        if easy_size == 'all':
            train_easy = lines[:len(lines)/2]
            test_easy = lines[len(lines)/2:]
        else:
            easy_size = int(easy_size)
            train_easy = lines[:easy_size]
            test_easy = lines[easy_size:easy_size+easy_size]
        train_svm = lines[:len(lines)/2]
        test_svm = lines[len(lines)/2:]

        file_name = "train_easy"
        save_file(file_name, train_easy)
        file_name = "train_svm"
        save_file(file_name, train_svm)
        file_name = "test_easy"
        save_file(file_name, test_easy)
        file_name = "test_svm"
        save_file(file_name, test_svm)
