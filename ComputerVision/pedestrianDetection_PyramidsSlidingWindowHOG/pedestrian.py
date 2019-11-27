import os
import platform
import math
import time
import random
import multiprocessing as mp

import cv2
import numpy as np
#import numba as nb
from sklearn import svm
from skimage import feature, exposure


def load_all(path):
    '''
        Load positive and negative images for training

        return:  lists of positive images, negative images, positive labels, negative labels
    '''
    print("Loading images...")
    _path = path + "/96X160H96/Train"
    files_pos = [os.path.join(_path + "/pos", f) for f in os.listdir(_path + "/pos")]
    _path = path + "/Train"
    files_neg = [os.path.join(_path + "/neg", f) for f in os.listdir(_path + "/neg")]
    images_pos = []
    images_neg = []
    labels_pos = []
    labels_neg = []
    if amount_MAX_IMAGES is not None:
        files_pos = files_pos[0:amount_MAX_IMAGES]
        files_neg = files_neg[0:amount_MAX_IMAGES]
    for image in files_pos:
        img = cv2.imread(image, 0)
        if platform.system() == 'Windows':
            lbl = image.split("\\")[1].split(".")[0]
        else:
            lbl = image.split("/")[5].split(".")[0]
        images_pos.append(img)
        labels_pos.append(lbl)
    for image in files_neg:
        img = cv2.imread(image, 0)
        if platform.system() == 'Windows':
            lbl = image.split("\\")[1].split(".")[0]
        else:
            lbl = image.split("/")[4].split(".")[0]
        images_neg.append(img)
        labels_neg.append(lbl)
    print("Positives: " + str(len(images_pos)))
    print("Negatives (x10): " + str(len(images_neg)))
    return images_pos, labels_pos, images_neg, labels_neg


def load_test(path, amount):
    '''
        Load images for testing

        return: lists of images, labels
    '''
    files = [os.path.join(path, f) for f in os.listdir(path)]
    files = files[0:amount]
    images = []
    labels = []
    for f in files:
        name = ""
        for char_idx in xrange(len(f)-1, 0, -1):
            char = f[char_idx]
            if platform.system() == 'Windows':
                if char == "\\":
                    label = name[::-1]
                    labels.append(label)
                    break
                else:
                    name += char
            else:
                if char == "/":
                    label = name[::-1]
                    labels.append(label)
                    break
                else:
                    name += char
        image = cv2.imread(f, 0)
        images.append(image)
    return images, labels


def load_notations(path):
    '''
        Read the files in the annotations folder

        return: list with all bounding boxes of each file
    '''
    files = [os.path.join(path, f) for f in os.listdir(path)]
    all_b_boxes = []
    for f in files:
        file_b_boxes = []
        with open(f, "r") as file_obj:
            for line in file_obj:
                if "filename" in line:
                    file_name = line
                    file_name = file_name.split("/")[2][:-2]
                if "Bounding" in line:
                    s = line.split(":")[1].split("-")
                    x_y_min = s[0][:-2][2:].split(", ")
                    x_y_max = s[1][:-2][2:].split(", ")
                    x_y_min = tuple([int(i) for i in tuple(x_y_min)])
                    x_y_max = tuple([int(i) for i in tuple(x_y_max)])
                    file_b_boxes.append((x_y_min, x_y_max))
        all_b_boxes.append((file_name, file_b_boxes))
    return all_b_boxes


def HOG_load(file_name, file_name_labels):
    hists = np.load(file_name)
    labels = np.load(file_name_labels)
    return hists.tolist(), labels.tolist()


def HOG_save(hists, labels, file_name, file_name_labels):
    hists_np = np.asarray(hists)
    labels_np = np.asarray(labels)
    np.save(file_name, hists_np, allow_pickle=False)
    np.save(file_name_labels, labels_np, allow_pickle=False)


def save_image(folder, label, image):
    '''
        Save an image in a folder,
        if the folder doesn't exist, it's created
    '''
    if not os.path.exists(folder):
        os.makedirs(folder)
    cv2.imwrite(folder + label + ".png", image)


def create_pyramids(path, amount, scale):
    '''
        Create pyramids for all images

        return: lists of pyramids, labels
    '''
    print("Creating pyramids...")
    images_test, labels_test = load_test(path, amount)
    images_pyramid = []
    all_scales = []
    for im in images_test:
        images, scales = pyramid(im, scale)
        images_pyramid.append(images)
        all_scales.append(scales)
    return images_pyramid, labels_test, all_scales


def pyramid(image, scale):
    '''
        Create pyramids for 1 image

        return: list of pyramids
    '''
    images = []
    resized_image = np.copy(image)
    w, h = np.shape(image)
    curr_size_h = h
    curr_size_w = w
    curr_scale = scale
    min_size_h = 128
    min_size_w = 128
    scales = []
    images.append(image)
    scales.append(1)
    while curr_size_h >= min_size_h and curr_size_w >= min_size_w: # downscale
        curr_size_w = int(w / curr_scale)
        curr_size_h = int(h / curr_scale)
        resized_image = cv2.resize(resized_image, (curr_size_h, curr_size_w))
        images.append(resized_image)
        scales.append(curr_scale)
        curr_scale += scale
        #save_image("pyramid/" + label + "/", label + "_" + str(curr_size_h), resized_image)       
    return images, scales


def sliding_window_process(q, svm_linear, pyramids, labels, scales):
    '''
        Slide the window through the image, function used by the processes

        return: list with all the positive bounding boxes, scales, and labels
    '''
    win_h = 128
    win_w = 64
    all_windows = []
    for pyr_idx in xrange(len(pyramids)):
        img_s_idx = 0
        pyr_windows = []
        def_image = np.copy(pyramids[pyr_idx][0])
        for image in pyramids[pyr_idx]:
            image_c = np.copy(image)
            curr_scale = scales[pyr_idx][img_s_idx]
            img_s_idx += 1
            for h in xrange(0, len(image) - win_h, 32):
                for w in xrange(0, len(image[h]) - win_w, 32):
                    window = image[h:h+win_h, w:w+win_w]
                    hist = HOG(window)
                    hist = hist.reshape(1, -1)
                    predict = svm_predict(svm_linear, hist)
                    if predict:
                        ini = (int(w * curr_scale), int(h * curr_scale))
                        end = (int((w + win_w) * curr_scale), int((h + win_h) * curr_scale))
                        cv2.rectangle(def_image, ini, end, (0,0,255), 2)
                        cv2.rectangle(image_c, (int(w), int(h)), (int(w + win_w), int(h + win_h)), (0,0,255), 2)
                        pyr_windows.append((ini, end))
            save_image("windows/" + str(labels[pyr_idx]) + "/", str(labels[pyr_idx]) + "_" + str(curr_scale), image_c)
        save_image("windows/", str(labels[pyr_idx]) + "_", def_image)
        all_windows.append((labels[pyr_idx], pyr_windows))
    q.put(all_windows)


def sliding_window(svm_linear, images_pyramid, labels_test, scales):
    '''
        Create processes and call the function to slide the window through the image

        Using Process because the standard python interpreter doesn't execute threads simultaneously

        The queue receives the bounding boxes from the processes
    '''
    b_boxes = []
    processes_amount = PROCESSES # must be < len(images_pyramid)
    processes = [None] * processes_amount
    qs = [None] * processes_amount
    print("Sliding windows... Pyramids: " + str(len(images_pyramid)))
    last_idx = (int(len(images_pyramid) / processes_amount) * processes_amount)
    start_chunck = 0
    chunck_size = int(len(images_pyramid) / processes_amount)
    end_chunck = chunck_size
    print("Creating " + str(processes_amount) + " processes...")
    for pss_idx in xrange(processes_amount):
        qs[pss_idx] = mp.Queue()
        processes[pss_idx] = mp.Process(target=sliding_window_process, args=(qs[pss_idx], svm_linear, images_pyramid[start_chunck:end_chunck], labels_test[start_chunck:end_chunck], scales[start_chunck:end_chunck]))
        processes[pss_idx].start()
        print("Process ID: " + str(processes[pss_idx].pid) + " for pyramids from " + str(start_chunck) + " to " + str(end_chunck))
        start_chunck = end_chunck
        end_chunck += chunck_size
    if last_idx < len(images_pyramid): # process with remaining pyramids in the list
        qs.append(mp.Queue())
        processes.append(mp.Process(target=sliding_window_process, args=(qs[processes_amount], svm_linear, images_pyramid[last_idx:], labels_test[last_idx:], scales[last_idx:])))
        processes[processes_amount].start()
        print("Process ID: " + str(processes[processes_amount].pid) + " with remaining pyramids, " + str(last_idx) + " to " + str(len(images_pyramid)))
    print("Waiting for processes to finish...")
    for pss_idx in xrange(len(processes)):
        b_b = qs[pss_idx].get()
        [b_boxes.append(b) for b in b_b]
        qs[pss_idx].close()
        qs[pss_idx].join_thread()
        processes[pss_idx].join()
    return b_boxes


def HOG(image):
    '''
        Histogram Of Gradients.
        the implemented one is commented at the end of the file

        return: list of histograms
    '''
    hist = feature.hog(image, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2))
    return hist


def HOG_check(images_pos, images_neg):
    '''
        Check if the histograms and labels files exist, if true load it, if false compute the histograms and save the files

        return: lists with histograms and label
    '''
    total = len(images_pos) + (len(images_neg) * 10)
    print("HOG...")    
    file_name = "HOGs.npy"
    file_name_labels = "labels.npy"
    hists = []
    labels = []
    if os.path.exists(file_name) and os.path.exists(file_name_labels):
        hists, labels = HOG_load(file_name, file_name_labels)
    if len(hists) != total or len(labels) != total:
        hists = []
        labels = []
        for i in xrange(len(images_pos)): # positive images
            image = images_pos[i]
            labels.append(1)        
            image_cut = image[16:144, 16:80]
            hists.append(HOG(image_cut))
        for i in xrange(len(images_neg)): # negative images
            image = images_neg[i]
            cuts = cut_negative(image)
            for ct in cuts:
                hists.append(HOG(ct))
                labels.append(0)
        HOG_save(hists, labels, file_name, file_name_labels)
    return hists, labels


def HNM(path, svm_linear, amount):
    '''
        Get 10 random small images (128, 64) from the negative images and test it to find false positives

        Tries to find 10k false positive images

        Using Process because the standard python interpreter doesn't execute threads simultaneously
        
        The queue receives the false positives from the processes

        return: list of false positive histograms
    '''
    print("Hard negative mining... Negative images: " + str(amount))
    test, _ = load_test(path, amount)
    hists_false_pos = []
    processes_amount = PROCESSES # must be < len(images_pyramid)
    processes = [None] * processes_amount
    qs = [None] * processes_amount
    last_idx = (int(len(test) / processes_amount) * processes_amount)    
    start_chunck = 0
    chunck_size = int(len(test) / processes_amount)
    end_chunck = chunck_size
    print("Creating " + str(processes_amount) + " processes...")
    for pss_idx in xrange(processes_amount):
        qs[pss_idx] = mp.Queue()
        processes[pss_idx] = mp.Process(target=HNM_process, args=(qs[pss_idx], test[start_chunck:end_chunck], svm_linear))
        processes[pss_idx].start()
        print("Process ID: " + str(processes[pss_idx].pid) + " for images from " + str(start_chunck) + " to " + str(end_chunck))
        start_chunck = end_chunck
        end_chunck += chunck_size
    if last_idx < len(test): # process with remaining images in the list
        qs.append(mp.Queue())
        processes.append(mp.Process(target=HNM_process, args=(qs[processes_amount], test[last_idx:], svm_linear)))
        processes[processes_amount].start()
        print("Process ID: " + str(processes[processes_amount].pid) + " with remaining images, " + str(last_idx) + " to " + str(len(test)))
    print("Waiting for processes to finish...")
    for pss_idx in xrange(len(processes)):
        hists = qs[pss_idx].get()
        [hists_false_pos.append(h) for h in hists]
        qs[pss_idx].close()
        qs[pss_idx].join_thread()
        processes[pss_idx].join()
    print("Adding more " + str(len(hists_false_pos)) + " false positives histograms")
    return hists_false_pos


def HNM_process(q, images, svm_linear):
    '''
        Hard Negative Mining function used by the processes

        Tries to find 10000/PROCESSES false positive images until the maximum interation amount

        It puts the false positives histograms in the queue to return to the main process
    '''
    hists_false_pos = []
    amount_neg_total = 10000 / PROCESSES
    limit = 0
    iterations = amount_HNM_ITERATIONS
    while (amount_neg_total > len(hists_false_pos)) and (limit < iterations):
        limit += 1
        for image in images:
            hists = []
            cuts = cut_negative(image)
            [hists.append(HOG(ct)) for ct in cuts]
            p = svm_predict(svm_linear, hists)
            [hists_false_pos.append(hists[x]) for x in xrange(len(p)) if p[x]]
            if amount_neg_total < len(hists_false_pos):
                break
    q.put(hists_false_pos)


def cut_negative(image):
    '''
        Cut 10 small images from each image

        return: list with images
    '''
    cuts = []
    small = len(image)
    if small > len(image[0]):
        small = len(image[0])
    for _ in xrange(0, 10):
        v = random.randint(1, small - 129)
        v_s = v + 128
        h = random.randint(1, small - 129)
        h_s = h + 64
        cuts.append(image[v:v_s, h:h_s])
    return cuts


def svm_train(hists, labels, c):
    '''
        Train the SVM using linear kernel

        return: SVM object
    '''
    print("Training SVM... C=" + str(c) + ", Histograms: " + str(len(hists)))
    svm_linear = svm.SVC(C=c, kernel='linear')
    svm_linear.fit(hists, labels)
    return svm_linear


def svm_predict(svm_linear, hists):
    '''
        Predict using the SVM

        return: list of predicted result
    '''
    prediction = svm_linear.predict(hists)
    return prediction


def svm_decision(svm_linear, hists):
    '''
        Compute the distace of each sample from the separating hyperplane

        return: list with distances
    '''
    dec = svm_linear.decision_function(hists)
    return dec


def exec_test(path, svm_linear, amount, pos_neg):
    '''
        Load and test positives and negatives images

        return: list of predicted result
    '''
    if pos_neg:
        print("Testing positives...")
        test, _ = load_test(path, amount)
        hists = []
        for image in test:
            cut = image[3:131, 3:67]
            hist = HOG(cut)
            hists.append(hist)
        return svm_predict(svm_linear, hists)
    else:
        print("Testing negatives...")        
        test, _ = load_test(path, amount)
        hists = []
        for image in test:
            cuts = cut_negative(image)
            [hists.append(HOG(ct)) for ct in cuts]
        return svm_predict(svm_linear, hists)


def test(svm_linear, amount):
    '''
        Run and show the test
    '''
    path_test = path + "/70X134H96/Test/pos"
    t = exec_test(path_test, svm_linear, amount, True)
    count = 0
    for x in t:
        if not x:
            count += 1
    print("False Negatives: " + str(count) + " out of " + str(amount))
    path_test = path + "/Test/neg"
    t = exec_test(path_test, svm_linear, amount, False)
    count = 0
    for x in t:
        if x:
            count += 1
    print("False Positives: " + str(count) + " out of " + str(amount))


def compare_b_boxes(ground_truths, b_boxes):
    for gt_box in ground_truths:
        for m_box in b_boxes: 
            if gt_box[0] == m_box[0]:
                for pos in gt_box[1]:
                    gt_x_y_min = pos[0]
                    gt_x_y_max = pos[1]
                    print(gt_x_y_min)
                    quit()



#pyramid_numba = nb.jit(nb.double[:,:](nb.double[:,:], nb.double))(pyramid)
#sliding_window_numba = nb.jit(nb.void(nb.double[:,:],nb.typeof(["a","a"]),nb.double[:,:]))(sliding_windows)

path = "./INRIAPerson"
amount_MAX_IMAGES = None # number of images to load, or None to load all
amount_TEST_IMAGES = 100 # number of images to test, false positives/false negatives
amount_B_B_IMAGES = 10 # number of images to test and draw the bounding boxes
amount_HNM_IMAGES = 1000 # number of images to send to the Hard Negative Mining
amount_HNM_ITERATIONS = 100 # number of iterations during the Hard Negative Mining, 10 = around 1k images when HNM_IMAGES = 1000
SCALE = 1.2 # scale to make the pyramids, must be > 1
C = 10000 # C parameter of SVM
PROCESSES = 4 # tested with 4, 6 and 8

if __name__ == "__main__":
    start = time.time()
    images_pos, labels_pos, images_neg, labels_neg = load_all(path)
    hists_all, labels_yes_no = HOG_check(images_pos, images_neg)
    svm_linear = svm_train(hists_all, labels_yes_no, C) # first training
    print("Test without hard negative mining:")
    test(svm_linear, amount_TEST_IMAGES)
    path_test = path + "/Train/neg"
    false_positives = HNM(path_test, svm_linear, amount_HNM_IMAGES) # hard negative mining
    for f_p in false_positives:
        hists_all.append(f_p)
        labels_yes_no.append(False)
    print("Train with hard negative mining:")
    svm_linear = svm_train(hists_all, labels_yes_no, C) # train with HNM result
    print("Test with hard negative mining:")
    test(svm_linear, amount_TEST_IMAGES)
    ''' sliding window '''
    path = "./INRIAPerson/Test/pos"
    images_pyramid, labels_test, scales = create_pyramids(path, amount_B_B_IMAGES, SCALE)
    b_boxes = sliding_window(svm_linear, images_pyramid, labels_test, scales)
    path = "./INRIAPerson/Test/annotations"
    ground_truths = load_notations(path)
    #compare_b_boxes(ground_truths, b_boxes)
    end = time.time()
    print("Execution time: " + str(end - start) + " seconds")


'''
implementacao do hog, porem com alguns erros
def Ax_Ay(image):
    mask_x = np.array([-1,0,1])
    mask_y = np.array([[-1,0,1]])
    Gx = cv2.filter2D(image, cv2.CV_32F, mask_x)
    Gy = cv2.filter2D(image, cv2.CV_32F, mask_y)
    magnitudes = np.zeros(np.shape(image))
    angles = np.zeros(np.shape(image))
    for c in xrange(len(Gx)):
        for l in xrange(len(Gx[c])):
            magnitudes[c, l] = math.sqrt((Gy[c, l] ** 2) + (Gx[c, l] ** 2))
            angles[c, l] = math.degrees(math.atan2(Gy[c, l], Gx[c, l]) * 0.5 + math.pi * 0.5)
    return angles, magnitudes
def normalize(hists):
    #for block in xrange(len(hists)):
    min = np.min(hists)
    max = np.max(hists)
    for b in xrange(len(hists)): 
        hists[b] = (hists[b] - min) / ((max - min) + 1e-5)
    return hists
def hog(image):
    angles, magnitudes = Ax_Ay(image)
    hists = []
    for l in xrange(0, len(image)-8, 8):
        for c in xrange(0, len(image[l])-8, 8):
            block_a = angles[l:l + 16, c:c + 16]
            block_m = magnitudes[l:l + 16, c:c + 16]
            for l_b in xrange(0, len(block_a), 8):
                for c_b in xrange(0, len(block_a[l_b]), 8): 
                    cell_a = block_a[l_b:l_b + 8, c_b:c_b + 8]
                    cell_m = block_m[l_b:l_b + 8, c_b:c_b + 8]
                    bins = [0] * 9
                    for l_c in xrange(len(cell_a)):
                        for c_c in xrange(len(cell_a[l_c])):
                            deg = cell_a[l_c, c_c]
                            pos = int(deg / 20)
                            p = ((deg / 20) - pos) * 100
                            if (deg / 20) - pos == 0.0: # 1 bin
                                if pos == 9:
                                    pos = 0
                                bins[pos] += cell_m[l_c, c_c]
                            else: # split 2 bins
                                bins[pos] += cell_m[l_c, c_c] * (100 - p) / 100
                                if pos == 8:
                                    pos = -1
                                bins[pos + 1] += cell_m[l_c, c_c] * p / 100
                    for b in bins:
                        hists.append(b)
    return hists
'''