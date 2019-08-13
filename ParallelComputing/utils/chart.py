import matplotlib.pyplot as plt
import csv
import sys

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Use: python " + str(__file__) + " chart.csv chart_persistent.csv") 
        quit()
    file_name = sys.argv[1]
    file_name_persistent = sys.argv[2]
    x = []
    y = []
    x2 = []
    y2 = []
    with open(file_name, "r") as file:
        r = csv.reader(file, delimiter=';')
        for row in r:
            x.append(int(row[0]))
            y.append(int(row[1]))
    with open(file_name_persistent, "r") as file:
        r = csv.reader(file, delimiter=';')
        for row in r:
            x2.append(int(row[0]))
            y2.append(int(row[1]))
    plt.xlabel("Vector size")
    plt.ylabel("Time")
    plt.plot(x, y, label='normal')
    plt.plot(x2, y2, label='persistent')
    plt.legend()
    plt.show()
