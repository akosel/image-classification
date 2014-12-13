
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import os
import sets
import sys
from PIL import Image
from termcolor import colored

from sklearn.decomposition import RandomizedPCA
from sklearn.neighbors import KNeighborsClassifier


# In[167]:

# setup a standard image size; this will distort some images but will get everything into the same shape
# functions based off of yhat's tutorial http://blog.yhathq.com/posts/image-classification-in-Python.html
def img_to_matrix(filename, useHsv=False, imageSize=(200, 200)):
    """
    takes a filename and turns it into a numpy array of RGB pixels
    optionally, include hsv values as well
    """
    img = Image.open(filename)
    img = img.resize(imageSize)
    if useHsv:
        rgb_img_data = list(img.getdata())
        hsv_img_data = list(img.convert('HSV').getdata())
        img = [rgb_img_data[idx] + hsv_img_data[idx] for idx in xrange(len(rgb_img_data))]
    else:
        img = list(img.getdata())
    img = map(list, img)
    img = np.array(img)
    return img

def flatten_image(img):
    """
    takes in an (m, n) numpy array and flattens it 
    into an array of shape (1, m * n)
    """
    s = img.shape[0] * img.shape[1]
    img_wide = img.reshape(1, s)
    return img_wide[0]

def getPictureList(path):
    fileList = []
    with open(path) as f:
        for row in f:
            fileList.append(row.strip() + '.jpg')
    return fileList

# image size, number of classes, number of components, number of k neighbors, isHsl
def printPrediction(imageSize, useHsv, numberOfClasses, numberOfComponents=5, numberOfKNeighbors=10):
    """
    takes a number of arguments and walks through the pipeline.
    does not return anything, but does write a file with the
    results in it.
    looks at existing files first to see if a particular prediction
    has already been run.
    """
    if cacheLookup(imageSize, useHsv, numberOfClasses, numberOfComponents, numberOfKNeighbors):
        return 'Found in cache'
    else:
        print 'Not found in cache, building results'
        
    images = [os.path.join('img', f) for f in getPictureList('not_bad.txt') if f[0] != '.']
    
    labels = [getLabel(f) for f in images]
    labelList = sorted(list(sets.Set(labels)))
    if numberOfClasses >= len(labelList):
      cutoffIdx = -1
    else:
      cutoffClass = labelList[numberOfClasses]
      cutoffIdx = labels.index(cutoffClass)

    images, labels = images[:cutoffIdx], labels[:cutoffIdx]
    labelList = sorted(list(sets.Set(labels)))
    data = []
    for image in images:
        img = img_to_matrix(image, useHsv=useHsv, imageSize=imageSize)
        img = flatten_image(img)
        data.append(img)

    data = np.array(data)
    
    is_train = np.random.uniform(0, 1, len(data)) <= 0.7
    y = np.array(labels)

    train_x, train_y = data[is_train], y[is_train]
    test_x, test_y = data[is_train==False], y[is_train==False]
    
    pca = RandomizedPCA(n_components = numberOfComponents)
    X = pca.fit_transform(data)
    
    knn = KNeighborsClassifier(n_neighbors = numberOfKNeighbors)
    knn.fit(train_x, train_y)

    ct = pd.crosstab(test_y, knn.predict(test_x), rownames=['Actual'], colnames=['Predicted'])
    filename = str(imageSize[0]) + '_' + str(imageSize[1]) + '_nco' + str(numberOfComponents) + '_ncl' + str(numberOfClasses) + '_nkn' + str(numberOfKNeighbors) + '.csv'
    ct.to_csv(os.path.join('results', filename))
    print ct
    
def cacheLookup(imageSize, useHsv, numberOfClasses, numberOfComponents, numberOfKNeighbors):
    """
    using the file naming convention defined in printPrediction,
    checks to see if the same classification has already run.
    """
    try:
        filename = str(imageSize[0]) + '_' + str(imageSize[1]) + '_nco' + str(numberOfComponents) + '_ncl' + str(numberOfClasses) + '_nkn' + str(numberOfKNeighbors) + '.csv'
        ct = pd.read_csv(os.path.join('results', filename))
        print ct
        return True
    except:
        return False
    
# helper functions
def getLabel(path):
    """
    small helper that takes a given filename, and returns
    the label, assuming the file is named like so:
    cornmarket_000079.jpg will return cornmarket
    """
    return path[path.index('/') + 1:path.rindex('_')]

def labelToInt(label):
    return labelList.index(label)

def intToLabel(idx):
    return labelList[idx]

if __name__ == '__main__':
  args = sys.argv[1:]
  if not args:
    print colored('To run this program, you need to pass five arguments on the command line. They are:', 'yellow')
    print colored('\nimage size, whether to include hsv, number of classes, number of components, and number of k neighbors', 'yellow')
  else:
    printPrediction((int(args[0]), int(args[0])), bool(args[1]), int(args[2]), int(args[3]), int(args[4]))
