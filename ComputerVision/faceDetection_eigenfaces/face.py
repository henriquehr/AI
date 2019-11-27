import cv2, os
import numpy as np
from PIL import Image
import math

cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)


def show_image(name, image):
    if SHOW_IMAGES:
        cv2.imshow(name, image.astype('uint8'))


def show_faces(eig_faces, h, w):
    for face in eig_faces:
        face = 255 * (face - np.min(face)) / (np.max(face) - np.min(face)) 
        show_image(str(face),  face.reshape(w, h))
 

def save_faces(eig_faces, mean_face, h, w, folder):
    if SAVE_FACES:
        for f in xrange(len(eig_faces)):
            eig_faces[f] = 255 * (eig_faces[f] - np.min(eig_faces[f])) / (np.max(eig_faces[f]) - np.min(eig_faces[f])) # normalize the values from 0 to 255 
            cv2.imwrite("faces/" + folder + "eig_face_"+str(f)+'.png', eig_faces[f].reshape(w, h))
        cv2.imwrite("faces/" + folder + "mean_face.png", np.resize(mean_face, (w, h)))


def normalize_images_yale(path):
    '''
        Load and normalize the images from the yale database

        return: matrix of images, height and width
    '''
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    labels = []
    expressions = []
    index = 0
    images_matrix = []
    w_max = 0
    h_max = 0
    for image_path in image_paths:
        image_pil = Image.open(image_path).convert('L')
        image = np.array(image_pil) # float
        label  = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))
        expression  = os.path.split(image_path)[1].split(".")[1]
        faces = faceCascade.detectMultiScale(image)
        for (x, y, w, h) in faces:
            images_matrix.append(image[y: y + h, x: x + w]) # ?
            labels.append(label)
            contains = False
            for e in expressions:
                if e[0] == expression:
                    e[1].append(index)
                    contains = True
                    break
            if not contains:
                expressions.append([expression, [index]])
            index += 1
            if w_max < w: # get the width and height
                w_max = w
            if h_max < h:
                h_max = h
    for i in xrange(len(images_matrix)):
        images_matrix[i] = cv2.resize(images_matrix[i], (h_max, w_max)).reshape(-1)
    return images_matrix, h_max, w_max, labels, expressions


def normalize_images_orl(path):
    '''
        Load and normalize the images from the orl database

        return: matrix of images, height and width
    '''
    sub_paths = [os.path.join(path, f) for f in os.listdir(path) if not os.path.isfile(os.path.join(path, f))] # if not f.endswith('README')
    images_path = []
    labels = []
    images_matrix = []
    w_max = 0
    h_max = 0
    for images_path in sub_paths:
        img = [os.path.join(images_path, f) for f in os.listdir(images_path)]
        label  = os.path.split(images_path)[1].split(".")[0]
        for image_path in img:
            image_pil = Image.open(image_path).convert('L')
            image = np.array(image_pil) # float
            images_matrix.append(image)
            labels.append(label)
            if w_max < np.shape(image)[0]:   
                w_max = np.shape(image)[0]
            if h_max < np.shape(image)[1]:
                h_max = np.shape(image)[1]
    for i in xrange(len(images_matrix)):
        images_matrix[i] = cv2.resize(images_matrix[i], (h_max, w_max)).reshape(-1)
    return images_matrix, h_max, w_max, labels


def remove_expression(images_matrix, labels, expressions, expression):
    '''
        Remove the expression from the images lists

        return: images_matrix and labels without the expression
    '''
    if expression == None:
        return images_matrix, labels
    else:
        removed = 0
        for e in expressions:
            if e[0] == expression:
                indices = e[1]
                for i in indices:
                    i -= removed
                    del labels[i]
                    del images_matrix[i]
                    removed += 1
                break
    return images_matrix, labels


def mean_face(images_matrix, h, w):
    '''
        Compute the mean face

        return: the mean face in (h_max * w_max) format
    '''
    sum_images = np.zeros((h * w)) # float
    for i in images_matrix:
        sum_images += i # sum all the images
    m_face = (sum_images / len(images_matrix)) # compute the mean_face  # (28900,)
    return m_face


def covariance(images_matrix, m_face):
    '''
        Compute the covariance matrix

        return: covariance matrix and covariace matrix - mean face
    '''
    n = images_matrix - m_face
    cov_mtx = np.cov(n)
    return cov_mtx, n


def eigen_faces(covariance, n):
    '''
        Compute the eigen values, eigen vectors and eigen faces

        return: eigen faces
    '''
    eig_values, eig_vectors = np.linalg.eig(covariance)
    eig_values = np.real(eig_values)
    eig_vectors = np.real(eig_vectors)
    indices = np.argsort(-eig_values)
    e_v_ordered = []
    for i in range(ef):
        e_v_ordered.append(eig_vectors[:,indices[i]])
    eig_faces = np.dot(e_v_ordered, n)
    # eig_values = eig_values[indices]
    # eig_vectors = eig_vectors[indices]
    return eig_faces


def knn(images_matrix, eig_faces, m_face, h, w, labels):
    '''
        Classify the images using the eigen faces

        knn algorithm (?)
    '''
    #print(">> classification knn() ")
    #print("classifying...")
    correct = 0
    sample_count = 0
    for sample in images_matrix: # sample image to test
        distances = []
        sample = np.dot(eig_faces, (sample - m_face))  # eigen faces * (image to sample - mean face)
        population_count = 0
        for population in images_matrix: # all the other images minus sample
            if sample_count != population_count:
                pop = np.dot(eig_faces, (population - m_face)) # eigen faces * (image to population - mean face)
                eu_dist = np.linalg.norm(sample - pop) # euclidean distance  # math.sqrt((math.pow(np.linalg.norm(sample - pop), 2)))
                distances.append((eu_dist, labels[population_count]))
            else:
                distances.append((float('Infinity'), labels[population_count])) # TODO (fix?) without this the distances list will be shifted to the left
            population_count += 1
        distances.sort()
        nn = distances[:K] # get the K nearest neighbors
        classes = [i[1] for i in nn]
        most_common_class = max(classes, key=classes.count) # get the most common face near the test image
        if most_common_class == labels[sample_count]:
            #print(">>Same person ")
            correct += 1
        else:
            pass
            #print(">>Different person ")
        sample_count += 1
    print("Correct: " + str(correct) + " out of " + str(len(images_matrix)) + " images")
    percent = correct * 100 / len(images_matrix)
    print("Accuracy: " + str(percent) + "%")
    return correct, percent


def k_fold_setup(images_matrix, labels):
    '''
        Separete the images in sub groups

        return: images and labels separeted in sub groups
    '''
    #print(">> k_fold_setup() ")
    if samples_amount > len(images_matrix) - 1:
        print("Sample size too high")
        quit()
        print("Quitting")        
    if float(float(len(images_matrix)) % float(samples_amount)) != 0:
        print("len(images_matrix) % samples_amount) must be 0")
        print("Quitting")
        quit()
    for i in xrange(len(images_matrix)):
        images_matrix[i] = np.append(images_matrix[i], (labels[i]))
    np.random.shuffle(images_matrix)
    sub_samples = [] # images_matrix[:len(images_matrix) / sample_size]
    last_pos = 0
    for s in range(samples_amount):
        last_pos = len(images_matrix) / samples_amount * (s + 1)
        sub_samples.append(images_matrix[(len(images_matrix) / samples_amount) * s:last_pos])
    group_labels = []
    for i in xrange(len(sub_samples)):
        sub_labels = []
        for j in xrange(len(sub_samples[i])):
            sub_labels.append(sub_samples[i][j][len(sub_samples[i][j]) - 1])
            sub_samples[i][j] = [float(a) for a in np.delete(sub_samples[i][j], len(sub_samples[i][j])-1)]
        group_labels.append(sub_labels)
    return sub_samples, group_labels


def k_fold(retained, retained_index, sub_samples, eig_faces, m_face, sub_labels):
    '''
        Split the images in groups of K samples

        return: Number of correct images and accuracy 
    '''
    #print(">> classification k_fold() ")
    correct = 0
    sample_count = 0
    for sample in retained: # sample image to test
        distances = []
        samp = np.dot(eig_faces, (sample - m_face))  # eigen faces * (image to sample - mean face)
        population_count = 0
        for population in sub_samples: # all the other images minus sample
            if sample_count != population_count:
                pop = np.dot(eig_faces, (population - m_face)) # eigen faces * (image to population - mean face)
                eu_dist = np.linalg.norm(samp - pop) # euclidean distance  # math.sqrt((math.pow(np.linalg.norm(sample - pop), 2)))
                distances.append((eu_dist, sub_labels[population_count]))
            else:
                distances.append((float('Infinity'), sub_labels[population_count])) # TODO (fix?) without this the distances list will be shifted to the left
            population_count += 1
        distances.sort()
        nn = distances[:K] # get the K nearest neighbors
        classes = [i[1] for i in nn]
        most_common_class = max(classes, key=classes.count) # get the most common face near the test image
        if most_common_class == sub_labels[sample_count]:
            #print(">>Same person ")
            correct += 1
        else:
            pass
            #print(">>Different person ")
        sample_count += 1
    #print("Correct: " + str(correct) + " out of " + str(len(sub_samples)) + " images")
    percent = correct * 100 / len(sub_samples)
    #print("Accuracy: " + str(percent) + "%")
    return correct, percent
                
def compute_stats(means, values, groups):
    sum_means = 0
    for m in xrange(len(means)):
        sum_means += math.sqrt(values[m] - means[m])
    variance = sum_means / len(groups)
    deviation = math.sqrt(variance)
    print("Standart deviation of all groups: " + str(deviation))
    return deviation


def compute(images_matrix, h, w, folder, labels, expressions, expression, samples_amount):
    def _compute_knn(images_matrix, h, w, folder, labels, expressions, expression): # helper function, computes knn
        images_matrix, labels = remove_expression(images_matrix, labels, expressions, expression)
        m_face = mean_face(images_matrix, h, w)
        cov, n = covariance(images_matrix, m_face)
        eig_faces =  eigen_faces(cov, n)
        correct, percent = knn(images_matrix, eig_faces, m_face, h, w, labels)
        save_faces(eig_faces, m_face, h, w, folder)
        show_faces(eig_faces, h, w)
        show_image("mean face", np.resize(m_face,(h,w)))            
        return expression, correct, percent
    if samples_amount == None: # k-fold test set to None
        if expression is 'ALL': # remove the face expressions one by one and make the average 
            results = []
            for e in expressions:
                print("Removed expression: " + str(e[0]))
                results.append(_compute_knn(list(images_matrix), h, w, folder, list(labels), expressions, e[0]))
            results.sort(key=lambda result: result[2],reverse=True)
            print("All images: ")
            all = _compute_knn(list(images_matrix), h, w, folder, list(labels), expressions, None)
            print("===============================================================================")
            print("Expression removed \tCorrect images \tAccuracy").expandtabs(30)
            for r in results:
                print(str(r[0]) +"\t"+ str(r[1]) + "\t"+ str(r[2])+"%" ).expandtabs(30)
            print("===============================================================================")
            print("No expression removed -> Correct images: " + str(all[1]) +"   Accuracy: "+ str(all[2])) + "%"
            print("===============================================================================")
        else: # test without one face expression
            _compute_knn(images_matrix, h, w, folder, labels, expressions, expression)
    else: # k-fold test
        sub_samples, sub_labels = k_fold_setup(list(images_matrix), list(labels))
        k_fold_results = []
        print(str(len(sub_samples)) + "-fold cross-validation results:  # all the images are randomly splitted in groups of equal size")
        print("Images in each group: " + str(len(sub_samples[0])))
        for retained_index in xrange(len(sub_samples)):
            retained = sub_samples[retained_index]
            for ss_index in xrange(len(sub_samples)):
                if retained_index != ss_index:
                    m_face = mean_face(sub_samples[ss_index], h, w)
                    cov, n = covariance(sub_samples[ss_index], m_face)
                    eig_faces =  eigen_faces(cov, n)
                    k_fold_results.append(k_fold(retained, retained_index, sub_samples[ss_index], eig_faces, m_face, sub_labels[ss_index]))
                    k_fold_results[len(k_fold_results) - 1] += (retained_index,) # tuple
                    save_faces(eig_faces, m_face, h, w, folder)
                    show_faces(eig_faces, h, w)
                    show_image("mean face", np.resize(m_face,(h,w)))
            retained_index += 1
        groups = []
        k_fold_results.sort(key=lambda result: result[2],reverse=False)
        index = 0
        for i in xrange(len(k_fold_results)):
            groups.append(k_fold_results[index:(len(sub_samples)-1) + index])
            index += len(sub_samples)-1
            if index >= len(k_fold_results):
                break
        total_i_c = 0
        total_a = 0
        means = []
        values = []
        for g in groups:
            print("Group retained " + str(g[0][2]) + ": \tCorrect images: \tAccuracy:").expandtabs(10)
            sum_i_c = 0
            sum_a = 0
            for r in g:
                print("\t       " + str(r[0]) + "\t    " + str(r[1]) + "%").expandtabs(22)
                sum_i_c += r[0]
                sum_a += r[1]
            total_i_c += sum_i_c
            total_a += sum_a
            values.append(total_i_c)
            means.append(total_i_c / len(g))
            print("Group average: \t" + str(sum_i_c / len(g)) + " images, \t" + str(sum_a / len(g)) + "%").expandtabs(0)
        tot = len(groups) * len(groups[0])
        print("Total: " + str(tot)  + " tests, average: " + str(total_i_c / tot) + ", accuracy: " + str(total_a / tot) + "%")
        compute_stats(means, values, groups)
        print("Testing all the images...")
        _compute_knn(images_matrix, h, w, folder, labels, expressions, None)
    if SHOW_IMAGES:
        cv2.waitKey()
        cv2.destroyAllWindows()


SHOW_IMAGES = False # show mean face and eigen faces, enable with caution
SAVE_FACES = True # save mean face and eigen faces in the 'faces' folder

'''
    Parameters to tune:
    ef: a positive number, tested with 5 and 10
    K: a positive number, tested with 1, 5 and 10
    expression: 'ALL', None, an expression/lighting name (that are in the database), ignored if samples_amount != None
    samples_amount: None, a positive number that matches the condition 'len(images_matrix) % samples_amount) == 0', tested with 10, 20, 40
    samples_amount must be None to use the expression variable
'''
ef = 5 # number of engen faces
K = 1 # knn
expression = 'ALL' # expression to remove, yale database only, 'ALL' to test all expressions one by one, None to test only 1 time with all the images
samples_amount = None # k-fold validation, if set, expression is ignored, None to use the leave-one-out
orl_only = False

print("Show images=" + str(SHOW_IMAGES))
print("Save faces=" + str(SAVE_FACES) + " # '/faces' folder")
print("Eigenfaces=" + str(ef))
print("K=" + str(K) + " # knn algorithm")
if not orl_only:
    print("")
    print(">>>>yale database...<<<<")
    print("Expression: " + str(expression))
    print("Samples (k-fold): " + str(samples_amount))
    path = './yale_faces'
    folder = "yale/"
    images_matrix, h, w, labels, expressions = normalize_images_yale(path)
    compute(images_matrix, h, w, folder, labels, expressions, expression, samples_amount)

expression = None # None for orl database
samples_amount = 10
expressions = None # None for orl database
path = './orl_faces'
folder = "orl/"
print("")
print(">>>>orl database...<<<<")
print("Expression: " + str(expression))
print("Samples (k-fold): " + str(samples_amount))
images_matrix, h, w, labels = normalize_images_orl(path)
compute(images_matrix, h, w, folder, labels, expressions, expression, samples_amount)
