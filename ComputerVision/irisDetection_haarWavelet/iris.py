import os
from threading import Thread
from collections import OrderedDict

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy import ndimage
from scipy import spatial
from sklearn.metrics import confusion_matrix
from skimage.feature import local_binary_pattern
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier


def load_images(results, index, files):
    '''
        Load the iris images, only usable if using threads
    '''
    images_ = []
    for file in files:
        images_.append(load_one_image(file))
    results[index] = images_ 


def load_one_image(file):
    file_loaded = cv2.imread(file, 0)
    if file_loaded is None:
        print("File not found: " + file)
        os._exit(-1)
    return file_loaded


def compute_eyes(images, feature, folder, thread_name, files_name):
    histog = []
    names = []
    for i in range(len(images)):
        name = files_name[i].split('/')
        name = name[len(name)-1]
        if COMPUTE_IRIS:
            pupil, edges, x, y, r = get_pupil(images[i])
            polar = None
            if pupil is not None:
                iris, polar = get_iris(pupil, x, y, r)
                names.append(name)
                histog.append(get_hist(polar,feature,folder,name))
                cv2.imwrite("iris/" + folder + "/" + name, polar)
            else:
                print("Thread_" + str(thread_name) + ": image(" + str(i) + ") " + name + " skipped. Pupil not found.")
                continue
        else:
            names.append(name)
            histog.append(get_hist(images[i],feature,folder,name))
        print("Thread_" + str(thread_name) + ": computed image(" + str(i) + "): " + name)
    histog = np.asarray(histog)
    names = np.asarray(names)
    file_name = "hist_" + str(feature) + "_" + str(folder)
    print("Thread_" + str(thread_name) + ": Saving histogram to file " + str(file_name))
    np.save(file_name, histog, allow_pickle=False)
    file_name = "names_" + str(feature) + "_" + str(folder)
    print("Thread_" + str(thread_name) + ": Saving images's names to file " + str(file_name))
    np.save(file_name, names, allow_pickle=False)
    print("Thread_" + str(thread_name) + " finished <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")


def get_hist(polar, feature, folder, name):
    img = None
    if feature == 'LBP':
        img = LBP(polar)
    elif feature == 'gabor':
        img = gabor(polar)
    elif feature == 'LoG':
        img = LoG(polar)
    cv2.imwrite(str(feature) + "/" + folder + "/" + name, img)
    hist, _ = np.histogram(img, bins=np.arange(49), range=(0, 48))
    eps=1e-7
    hist = hist.astype("float")
    hist /= (hist.sum() + eps)
    return np.asarray(hist)


def get_iris_code(image):
    cA4, (cH4, cV4, cD4), (cH3, cV3, cD3), (cH2, cV2, cD2), (cH1, cV1, cD1) = pywt.wavedec2(image, 'haar', level=4)
    x, y = cA4.shape
    iris_code = np.zeros((x, y), dtype=int)
    for i in xrange(len(cA4[:,0])):
        for j in xrange(len(cA4[0,:])):
            if (cH4[i,j] + cV4[i,j] + cD4[i,j]) >= 0:
                iris_code[i,j] = 1
            else:
                iris_code[i,j] = 0
    return iris_code


def hamming_distance(u, v):
    dist = spatial.distance.hamming(u, v)
    return dist


def compute_FAR_FRR(images_left, images_right, files_left, files_right):
    print("Computing FAR/FRR")
    images = np.append(images_left, images_right, axis=0)
    labels = np.append(files_left, files_right, axis=0)
    labels = [f.split("/")[2] for f in labels]
    false_positive = 0.0  # false positive, different class recognized as the same class
    false_negative = 0.0  # false negative, same class recognized as different class
    true_positive = 0.0  # true positive, same class correctly recognized
    true_negative = 0.0 # true negative, different class correctly not recognized
    threshold = 0.97
    print("Images: " + str(len(images))) +" ("+ str(len(images)/2) + " left and " + str(len(images)/2) + " right)"
    print("Loop size is around: " + str(((len(images)*len(images))/2)-(len(images)/2)))
    print("Threshold: " + str(threshold))
    print("please wait...")
    for i_1 in xrange(len(images)):
        current = labels[i_1][2:5]
        for i_2 in xrange(i_1+1, len(images)): # don't consider the same image or an image already tested
            h_d = hamming_distance(images[i_1].flatten(), images[i_2].flatten())
            test_class = labels[i_2][2:5]
            if h_d <= threshold:
                if current == test_class:
                    true_positive += 1.0
                else:
                    false_positive += 1.0
            else:
                if current == test_class:
                    false_negative += 1.0
                else:
                    true_negative += 1.0

    print("False Positives: " + str(int(false_positive)))
    print("True Positives: " + str(int(true_positive)))
    print("False Negatives: " + str(int(false_negative)))
    print("True Negatives: " + str(int(true_negative)))

    print("Accuracy: " + str((true_positive + true_negative) / (true_positive+true_negative+false_positive+false_negative)))

    far = false_positive / (false_positive + true_negative)
    frr = false_negative / (true_positive + false_negative)
    print("FAR: " + str(far))
    print("FRR: " + str(frr))


def LoG(image):
    g_l = ndimage.filters.gaussian_laplace(image, 2, mode='reflect')
    return g_l


def gabor(image):
    filters = []
    for theta in np.arange(0, np.pi, np.pi / 32):
        kern = cv2.getGaborKernel((12, 12), 8.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        kern /= 1.5*kern.sum()
        filters.append(kern)
    accum = np.zeros_like(image)
    for kern in filters:
        filtered = cv2.filter2D(image, cv2.CV_8UC3, kern)
        np.maximum(accum, filtered, accum)
    return accum


def LBP(image):
    return local_binary_pattern(image, 8*3, 3)
    '''height, width = np.shape(image)
    lbp_img = np.zeros((height,width))
    x = 1
    y = 1
    bin_list = [1, 2, 4, 8, 16, 32, 64, 128]
    for h in xrange(1, height - 2):
        for w in xrange(1, width - 2):
            values = []
            center = image[x+h][y+w]
            values.append(1 if image[x-1+h][y+1+w] >= center else 0) #top left
            values.append(1 if image[x+h][y+1+w] >= center else 0) #top
            values.append(1 if image[x+1+h][y+1+w]>= center else 0) #top right
            values.append(1 if image[x+1+h][y+w] >= center else 0) #right
            values.append(1 if image[x+1+h][y-1+w] >= center else 0) # bottom right
            values.append(1 if image[x+h][y-1+w] >= center else 0) #bottom
            values.append(1 if image[x-1+h][y-1+w] >= center else 0) # bottom left
            values.append(1 if image[x-1+h][y+w] >= center else 0) #left
            val = 0
            for c in xrange(len(values)):
                val += values[c] * bin_list[c]
            lbp_img[h][w] = val
    return lbp_img'''


def get_iris(image, x, y, r):
    r += 3
    r_orig = r
    result  = image.copy()    
    colors = []
    coords = []
    image = cv2.equalizeHist(image)       
    x_R = 0
    for i in range(30,90):
        point_r = np.zeros((image.shape[0],image.shape[1]),np.uint8)
        point_l = np.zeros((image.shape[0],image.shape[1]),np.uint8)
        x_R = r+i
        coords.append(x_R)
        '''
            Instead of using circles, using rectangles shifting horizontally, to reduce the noise from the other parts of the eye
        '''
        cv2.rectangle(point_r,(x+r+i-2,y-11), (x+r+i,y+11),(255,0,23),thickness=1) 
        cv2.rectangle(point_l,(x-r-i-2,y-11), (x-r-i,y+11),(255,0,23),thickness=1)
        masked_data_1 = cv2.bitwise_and(image, image, mask=point_r)
        masked_data_2 = cv2.bitwise_and(image, image, mask=point_l)       
        points = cv2.bitwise_xor(masked_data_1, masked_data_2)
        avg_color_per_row = cv2.mean(points)
        avg_color = cv2.mean(avg_color_per_row)
        colors.append(avg_color[0])
        #img = cv2.bitwise_xor(image, points)
        #stack = np.hstack((img,points))
        #cv2.imshow("circles", stack)
        #cv2.waitKey()
    '''
        Find the outer border of the iris probably in the worst possible way
    '''
    '''past = 0
    now = 0
    save = []
    save_idx = []
    for i in xrange(len(colors) - 1):
        c_1 = colors[i]
        c_2 = colors[i+1]
        now += c_1 + c_2
        if now < past:
            save.append(now)
            save_idx.append(i)
            r = coords[i]
        past = now
        now = 0
    max = 0
    max_idx = 0
    for i in xrange(len(save) - 1):
        c = save[i] - save[i-1]
        if max < c:
            max = c
            max_idx = save_idx[i]
    r = coords[max_idx] '''
    r = coords[55] # fixed size because of hamming distance
    '''
        Extract the iris
    '''
    mask = np.zeros((result.shape[0],result.shape[1]),np.uint8)
    cv2.circle(mask, (x, y),(r), (255,255,255), r_orig - r)
    result = cv2.bitwise_and(result, result, mask=mask)
    nsamples = 360
    samples = np.linspace(0,2.0 * np.pi, nsamples)[:-1]
    r_start = r - r_orig
    polar = np.zeros((r_start, nsamples))
    for i_r in range(r_start):
        for t in samples:
            x_pos = (i_r + r_orig) * np.cos(t) + x
            y_pos = (i_r + r_orig) * np.sin(t) + y
            if x_pos < result.shape[1] and y_pos < result.shape[0]:
                polar[int(i_r)][int(t * nsamples / 2.0 / np.pi)] = result[int(y_pos)][int(x_pos)]
            else:
                polar[int(i_r)][int(t * nsamples / 2.0 / np.pi)] = 0
    return result, polar


def get_pupil(image):
    #original = image.copy()
    img = image.copy()
    img = cv2.GaussianBlur(img, (9,9),0)
    img = cv2.medianBlur(img, 3)
    ret, img = cv2.threshold(img,25,255,cv2.THRESH_BINARY)
 
    kernel = np.ones((5,5),np.uint8)
    img = cv2.erode(img,kernel,iterations = 8)
    img = cv2.dilate(img,kernel,iterations = 8)
    img = cv2.Canny(img,140,180)
    #cv2.imshow("1", image)
    #cv2.waitKey()

    edges = img
    method = ''
    if cv2.__version__.split('.')[0] < '3':
        method = cv2.cv.CV_HOUGH_GRADIENT
    else:
        method = cv2.HOUGH_GRADIENT
    circles = cv2.HoughCircles(img, method, 3.1, 200, param1=180, param2=120, minRadius=1, maxRadius=120)
    if circles is None:
        #print("No circle found")
        return None, None, 0, 0, 0
    for (x,y,r) in circles[0,:]:
        cv2.circle(image, (x,y), r, (0,0,0), -1)
    x, y, r = circles[0][0].astype('int')
    return image, edges, x, y, r


def load_histograms(file_1, file_2, files_names_1, files_names_2):
    content_1 = np.load(file_1)
    content_2 = np.load(file_2)
    data = np.append(content_1, content_2, axis=0)
    names_1 = np.load(files_names_1)
    names_2 = np.load(files_names_2)
    labels = np.append(names_1, names_2)
    return data, labels


def classify_kfold(data, labels, clsfier):
    print("Classifying, please wait...")
    data = np.asarray(data)
    labels = np.asarray(labels)
    print("Number of images: " + str(len(data)))
    print("Number of labels: " + str(len(labels)))
    def _t(train_i, test_i, s_index):
            train_d, test_d = data[train_i], data[test_i]
            train_l, test_l = labels[train_i], labels[test_i]
            if clsfier == 'svm':
                _c = svm.SVC(kernel='rbf', gamma=0.001, C=10)
            elif clsfier == 'knn':
                _c = KNeighborsClassifier(5)
            _c.fit(train_d, train_l)
            predictions[s_index] = _c.score(test_d, test_l) * 100
    K = 5 # kfold number, this is also the number of threads
    kf = KFold(n_splits=K, shuffle=False)
    predictions = [None] * K
    threads = [None] * K
    i = 0
    for train_i, test_i in kf.split(data):
        #_t(train_i, test_i, i)
        threads[i] = Thread(target=_t, args=(train_i, test_i, i))
        threads[i].start()
        i += 1
    for i in xrange(K):
        threads[i].join()
    print("Score: " + str(predictions))
    print("Por algum motivo da 0.0, nao consegui descobrir o por que.")


def load_iris(files_left, files_right):
    curr_folder = [os.path.join('iris', f) for f in os.listdir('iris')]
    curr_files = os.listdir(curr_folder[0])
    curr_files.sort()
    for _f in curr_files:
        if(_f.endswith('.jpg')):
            files_left.append(os.path.join(curr_folder[0], _f))
    curr_files = os.listdir(curr_folder[1])
    curr_files.sort()
    for _f in curr_files:
        if(_f.endswith('.jpg')):
            files_right.append(os.path.join(curr_folder[1], _f))



iris_dir = "CASIA-Iris-Lamp-100"
#iris_dir_2 = "CASIA-IrisV4-Interval"

''' 
    Params to run:
        1: FAR_FRR = False, CLASSIFICATION_ONLY = False, COMPUTE_IRIS = True, LIMIT_IMAGES = a number or None (None = all the images)
        2: FAR_FRR = False, CLASSIFICATION_ONLY = True, COMPUTE_IRIS = False, LIMIT_IMAGES = same as 1
        3: FAR_FRR = True, CLASSIFICATION_ONLY = False, COMPUTE_IRIS = False, LIMIT_IMAGES = same as 1
'''
''' The priority is: FAR_FRR > CLASSIFICATION_ONLY > COMPUTE_IRIS '''
FEATURE = 'LBP' # 'LBP' 'gabor' 'LoG'
FAR_FRR = True # if true, the algorithm will compute FAR/FRR and exit
CLASSIFICATION_ONLY = False # if true, the algorithm will classify the images and exit
COMPUTE_IRIS = False # if the iris images were computed before, you can load it from the hdd instead of computing again
LIMIT_IMAGES = 200 # limit the number of images to use or None to use all the images

ONE_THREAD = False # for code testing only

files_left = []
files_right = []

if FAR_FRR:
    load_iris(files_left, files_right)
    if LIMIT_IMAGES is not None:
        files_left = files_left[:LIMIT_IMAGES]
        files_right = files_right[:LIMIT_IMAGES]
    threads_amount = 2
    threads = [None] * threads_amount
    results = [None] * threads_amount
    threads[0] = Thread(target=load_images, args=(results, 0, files_left))
    threads[1] = Thread(target=load_images, args=(results, 1, files_right))
    threads[0].start()
    threads[1].start()
    threads[0].join()
    threads[1].join()
    images_left = results[0]
    images_right = results[1]
    compute_FAR_FRR(images_left, images_right, files_left, files_right)
    os._exit(-1)

if CLASSIFICATION_ONLY:
    file_1 = "hist_"+FEATURE+"_L.npy"
    file_2 = "hist_"+FEATURE+"_R.npy"    
    file_names_1 = "names_"+FEATURE+"_L.npy"
    file_names_2 = "names_"+FEATURE+"_R.npy"
    data, labels = load_histograms(file_1, file_2, file_names_1, file_names_2)
    clsfier = 'svm' # 'svm' 'knn'
    classify_kfold(data, labels, clsfier)
    os._exit(-1)
if COMPUTE_IRIS:
    folders = [os.path.join(iris_dir, f) for f in os.listdir(iris_dir)]    
    folders.sort()
    for folder in folders:
        curr_folder = [os.path.join(folder, f) for f in os.listdir(folder)]
        curr_files = os.listdir(curr_folder[0])
        curr_files.sort()
        for _f in curr_files:
            if(_f.endswith('.jpg')):
                files_left.append(os.path.join(curr_folder[0], _f))
        curr_files = os.listdir(curr_folder[1])
        curr_files.sort()
        for _f in curr_files:
            if(_f.endswith('.jpg')):
                files_right.append(os.path.join(curr_folder[1], _f))
else:
    load_iris(files_left, files_right)

if LIMIT_IMAGES is not None:
    files_left = files_left[:LIMIT_IMAGES]
    files_right = files_right[:LIMIT_IMAGES]

if not ONE_THREAD:
    threads_amount = 2
    threads = [None] * threads_amount
    results = [None] * threads_amount
    threads[0] = Thread(target=load_images, args=(results, 0, files_left))
    threads[1] = Thread(target=load_images, args=(results, 1, files_right))
    threads[0].start()
    threads[1].start()

    for i in range(len(threads)):
        threads[i].join()

    images_left = results[0]
    images_right = results[1]

    threads[0] = Thread(target=compute_eyes, args=((images_left,FEATURE,"L",0, files_left)))
    threads[1] = Thread(target=compute_eyes, args=((images_right,FEATURE,"R",1, files_right)))
    threads[0].start()
    threads[1].start()

    for i in range(len(threads)):
        threads[i].join()
else:
    results = [None] * 1
    load_images(results, 0, files_left)
    images_left = results[0]
    compute_eyes(images_left,FEATURE,"L",0, files_left)
