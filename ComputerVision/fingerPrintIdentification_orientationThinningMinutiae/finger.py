import os
import math
import platform

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy import signal
from skimage.morphology import skeletonize


def load(path, type):
    print("Database: " + str(path))
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(type)]
    images = []
    if MAX_IMAGES is not None:
        image_paths = image_paths[0:MAX_IMAGES]
    for image in image_paths:
        if type == "jpg":
            file = cv2.imread(image, 0)
        elif type == "raw":
            file = np.fromfile(image, dtype=np.uint8)
        file = np.reshape(file, (300, 300))
        images.append(file)
    if platform.system() == 'Windows':
        labels = [f.split("\\")[1].split(".")[0] for f in image_paths] # windows
    else:
        labels = [f.split("/")[2].split(".")[0] for f in image_paths] # linux
    print("Loaded " + str(len(images)) + " images")
    return images, labels


def save_image(folder, label, image):
    cv2.imwrite(folder + label + ".jpg", image)


def enhancement(image, label):
    print("Enhancement...")
    image_enhanced = np.zeros(np.shape(image), 'float32')
    u = np.mean(image)
    s = np.std(image)
    for l in xrange(len(image)):
        for c in xrange(len(image[l])):
            tmp = 150. + 95. * (float(image[l][c] - u) / s)
            image_enhanced[l, c] = tmp
            if tmp > 255:
                image_enhanced[l, c] = 255
            elif tmp < 0:
                image_enhanced[l, c] = 0
    save_image("enhanced/", label, image_enhanced)
    return image_enhanced


def Ax_Ay(image):
    image_filtered = ndimage.median_filter(image, 5) # filter 5x5
    Gx = cv2.Sobel(image_filtered, cv2.CV_64F, 1, 0) # sobel x
    Gy = cv2.Sobel(image_filtered, cv2.CV_64F, 0, 1) # sobel y
    Ax = np.zeros(np.shape(image))
    Ay = np.zeros(np.shape(image))
    for l in xrange(0, len(image)):
        for c in xrange(0, len(image[l])):
            Ax[l][c] = Gx[l][c] ** 2 - Gy[l][c] ** 2  # Gx^2 - Gy^2
            Ay[l][c] = 2 * Gx[l][c] * Gy[l][c] # 2 * Gx * Gy
    return Ax, Ay


def orientation(image, label):
    print("Orientation...")
    angles = np.zeros(np.shape(image))
    Ax, Ay = Ax_Ay(image)
    w = 11 # block size
    image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    for l in xrange(0, len(image), w): # 11x11
        for c in xrange(0, len(image[l]), w):
            _Ax = Ax[l:l + w, c:c + w].sum() / math.pow(w, 2) # sum of pixels of each block / w^2
            _Ay = Ay[l:l + w, c:c + w].sum() / math.pow(w, 2)
            ang = math.atan2(_Ay, _Ax) * 0.5 + math.pi * 0.5
            start_x = l + w * 0.5 # center
            start_y = c + w * 0.5
            length = w * 0.5
            # draw a line knowing its x,y and angle [(length * sin||cos) + x||y]
            end_x = length * math.sin(ang) + start_x
            end_y = length * math.cos(ang) + start_y
            cv2.line(image_color, (int(start_y), int(start_x)), (int(end_y), int(end_x)), (0, 0, 255)) # opencv swap x and y
    save_image("orientation/", label, image_color)


def singular_point(image, label):
    print("Singular point...")
    h, v = np.shape(image)
    angles = np.zeros((int(h / 11), int(v / 11)))
    a = np.zeros((int(h / 11), int(v / 11)))
    b = np.zeros((int(h / 11), int(v / 11)))
    images_Ax = np.zeros((int(h / 11), int(v / 11)))
    images_Ay = np.zeros((int(h / 11), int(v / 11)))
    filter = np.array([[1,1,1],[1,2,1],[1,1,1]])
    idx_Ax = 0
    idx_Ay = 0
    image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)    
    Ax, Ay = Ax_Ay(image)
    for l in xrange(0, v - 11, 11): # 11x11
        for c in xrange(0, h - 11, 11):
            _Ax = Ax[l:l + 11, c:c + 11].sum() / math.pow(11, 2)
            _Ay = Ay[l:l + 11, c:c + 11].sum() / math.pow(11, 2)
            images_Ax[idx_Ax, idx_Ay] = _Ax
            images_Ay[idx_Ax, idx_Ay] = _Ay
            idx_Ay += 1
        idx_Ax += 1
        idx_Ay = 0
    a = signal.convolve(images_Ax, filter, 'same')
    b = signal.convolve(images_Ay, filter, 'same')
    for l in xrange(0, int(v / 11)):
        for c in xrange(0, int(h / 11)):
            angles[l, c] = math.atan2(b[l, c], a[l, c]) * 0.5 + math.pi * 0.5
    for l in xrange(0, len(angles) - 3):
        for c in xrange(0, len(angles) - 3):
            if np.mean(image[l * 11:(l * 11) + 11, c * 11:(c * 11) + 11]) != 255:
                block_curr = angles[l:l + 3, c:c + 3].flatten()
                block_curr = np.delete(block_curr, 5)
                s = 0
                _s = 0
                for l_b in xrange(len(block_curr) - 1):
                    s = math.degrees(block_curr[l_b] - block_curr[l_b + 1])
                    if s >= 90:
                        _s -= 180
                    elif s <= -90:
                        _s += 180
                s = math.degrees(block_curr[7] - block_curr[0])
                if s >= 90:
                    _s -= 180
                elif s <= -90:
                    _s += 180
                if _s == 180:
                    #print("Core")
                    cv2.circle(image_color, (c * 11, l * 11), 5, (0, 0, 255), 2)
                elif _s == -180:
                    #print("Delta")
                    cv2.circle(image_color, (c * 11, l * 11), 5, (0, 255, 0), 2)
    save_image("singular_point/", label, image_color)


def roi(image, label):
    print("ROI...")
    image_roi = np.copy(image)
    w = 11
    image_center = np.array((150,150))
    corner = np.array((0,0))
    max_dist = image_center - corner
    i_max = image.max()
    i_min = image.min()
    i_mean = []
    i_std = []
    for l in xrange(0, len(image), w):
        for c in xrange(0, len(image[l]), w):
            curr_block = np.array(image[l:l + w, c:c + w])
            i_mean.append(np.mean(curr_block))
            i_std.append(np.std(curr_block))
    max_mean = np.max(i_mean)
    max_std = np.max(i_std)
    min_mean = np.min(i_mean)
    min_std = np.min(i_std)
    for l in xrange(0, len(image), w):
        for c in xrange(0, len(image[l]), w):
            curr_block = np.array(image[l:l + w, c:c + w], 'float')
            curr_block = (curr_block - i_min) / (i_max - i_min)
            block_center = np.array((l + (w / 2), c + (w / 2)))
            ratio = 1 - np.linalg.norm(block_center - image_center) / np.linalg.norm(max_dist)
            mean = np.mean(curr_block) - min_mean / max_mean
            std = np.std(curr_block) - min_std / max_std
            v = 0.5 * (1 - mean) + 0.5 * std + ratio
            if v <= 0.8:
                image_roi[l:l + w, c:c + w] = 255
    save_image("roi/", label, image_roi)
    return image_roi


def binarization(image, label):
    print("Binarization...")
    image_tmp = np.zeros(np.shape(image), 'float32')
    '''for c in xrange(len(image)):
        for l in xrange(len(image[c])):
            p = image[c, l]
            image_tmp[c, l] = 0 if p < 127 else 255
    cv2.imwrite("binary/" + label + ".jpg", image_tmp)
    return image_tmp'''
    hist, _ = np.histogram(image, bins=np.arange(255))
    sum_1 = 0
    sum_2 = 0
    i_1 = 0
    i_2 = 0
    for i in xrange(len(hist)):
        if sum_1 <= 90000 / 4:
            sum_1 += hist[i]
            i_1 = i
        if sum_2 <= 90000 / 2:
            sum_2 += hist[i]
            i_2 = i
        else:
            break
    P2 = i_1
    P5 = i_2
    b_s = 3
    for l in xrange(0, len(image)-3, b_s):
        for c in xrange(0, len(image[l])-3, b_s):
            curr_block = image[l:l + b_s, c:c + b_s]    
            M = np.mean(image[l:l + 11, c:c + 11])
            #inner_block = curr_block[4:7, 4:7]
            inner_block = curr_block
            S = 0
            if len(inner_block) == 3 and len(inner_block[0]) == 3:
                S = (np.sum(inner_block) - inner_block[1, 1]) / 8
            for l_b in xrange(len(curr_block)):
                for c_b in xrange(len(curr_block[l_b])):
                    if curr_block[l_b, c_b] < P2:
                        curr_block[l_b, c_b] = 0
                    elif curr_block[l_b, c_b] > P5:
                        curr_block[l_b, c_b] = 255
                    else:
                        curr_block[l_b, c_b] = 255 if S >= M else 0
            image_tmp[l:l + b_s, c:c + b_s] = curr_block
    save_image("binary/", label, image_tmp)
    return image_tmp


def smoothing(image, label):
    print("Smoothing...")
    image_smooth = np.zeros(np.shape(image), 'float32')
    f = 5
    center = int(f/2)
    for l in xrange(0, len(image) - f):
        for c in xrange(0, len(image[l]) - f):
            curr_block = image[l:l + f, c:c + f]
            if np.count_nonzero(curr_block) >= 18:
                image_smooth[l + center, c + center] = 255
            elif (25 - np.count_nonzero(curr_block)) >= 18:
                image_smooth[l + center, c + center] = 0
            else:
                image_smooth[l + center, c + center] = curr_block[center, center]
    f = 3
    center = int(f/2)
    for l in xrange(0, len(image) - f):
        for c in xrange(0, len(image[l]) - f):
            curr_block = image[l:l + f, c:c + f]
            if np.count_nonzero(curr_block) >= 5:
                image_smooth[l + center, c + center] = 255
            elif (9 - np.count_nonzero(curr_block)) >= 5:
                image_smooth[l + center, c + center] = 0
            else:
                image_smooth[l + center, c + center] = curr_block[center, center]
    save_image("smooth/", label, image_smooth)    
    return image_smooth


def thinning(image, label):
    print("Thinning...")
    image_tmp = np.zeros(np.shape(image))
    _, _image = cv2.threshold(image, 250, 255, cv2.THRESH_BINARY)
    for l in xrange(len(_image)):
        for c in xrange(len(_image[l])):
            image_tmp[l, c] = 0 if _image[l, c] > 0 else 1
    skeleton = skeletonize(image_tmp)
    for l in xrange(len(skeleton)):
        for c in xrange(len(skeleton[l])):
            image_tmp[l, c] = 255 if skeleton[l, c] == 0 else 0
    save_image("thin/", label, image_tmp)    
    return image_tmp


def minutae(image, label):
    print("Minutae...")
    image_tmp = np.array(image, 'float32')
    image_color = cv2.cvtColor(image_tmp, cv2.COLOR_GRAY2RGB)
    positions = []
    store = []
    for c in xrange(len(image) - 3):
        for l in xrange(len(image[c]) - 3):
            curr_block = image[l:l + 3, c:c + 3]
            if np.mean(curr_block) == 255:
                continue
            center = (c + 1, l + 1)
            p = 0
            if curr_block[1, 1] < 255:
                for c_b in xrange(len(curr_block)):
                    for l_b in xrange(len(curr_block[c_b])):
                        if curr_block[c_b, l_b] < 255:
                            p += 1
                            store.append(center)
            else:
                continue
            if p == 2:
                positions.append(("ending", center))
                cv2.circle(image_color, center, 2, (255, 0, 0), -1)
            elif p == 4:
                positions.append(("bifurcation", center))
                cv2.circle(image_color, center, 2, (0, 0, 255), -1)
            #elif p == 5:
                #positions.append(("crossing", center))
                cv2.circle(image_color, center, 2, (255, 0, 0), -1)
    #for pos in xrange(len(positions)):
    #        for idx in xrange(len(store)):
    #            dist = math.sqrt((positions[pos][1][0] - store[idx][0]) ** 2 + (positions[pos][1][1] - store[idx][1]) ** 2)
    #            #print(dist)
    #            if dist == 0:
    #                cv2.circle(image_color, positions[pos][1], 2, (0, 0, 255), -1)
    #            elif dist > 200:
    #                cv2.circle(image_color, positions[pos][1], 2, (0, 0, 255), -1)
    save_image("minutae/", label, image_color)


#path = "./Lindex101"
path = "./Rindex28"

MAX_IMAGES = 10

images, labels = load(path, "raw")
for i in range(len(images)):
    image = images[i]
    label = labels[i]
    print("Image: " + str(i) + " - " + str(label))
    image_enhanced = enhancement(image, label)
    orientation(image_enhanced, label)
    image_roi = roi(image_enhanced, label)
    singular_point(image_roi, label)
    image_binary = binarization(image_roi, label)
    image_smooth = smoothing(image_binary, label)
    image_thin = thinning(image_smooth, label)
    minutae(image_thin, label)