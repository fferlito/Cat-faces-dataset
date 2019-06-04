# import the necessary packages
import cv2
import glob
import numpy as np
import imutils

path = "/home/federico/Desktop/Deep learning/cat-generator/cat face detector/coco-dataset"
path_aug = "/home/federico/Desktop/Deep learning/cat-generator/cat face detector/coco-modified-dataset"


# save all of the file names to a list and
# then loop through this list reading your images (numpy arrays) into a new list
print("\nRetrieving cats images from:")
print(path)
folders = glob.glob(path)
i = 0
imagenames = []
for folder in folders:
    for f in glob.glob(folder+'/*.jpg'):
        i = i +1
        imagenames.append(f)

print("\nCat images found: " + np.str(i))

detector = cv2.CascadeClassifier("/home/federico/Desktop/Deep learning/cat-generator/haarcascade_frontalcatface.xml")
j = 1

print("\nSaving cat faces in:")
print(path_aug)
for image in imagenames:
    # open image
    img = cv2.imread(image)
    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except Exception as e:
        print(str(e))
    # The detectMultiScale  function returns rects , a list of 4-tuples.
    # These tuples contain the (x, y)-coordinates and width and height of each detected cat face.
    if not (img is None):
        print(image)
        rects = detector.detectMultiScale(img, scaleFactor=1.02, minNeighbors=2, minSize=(64, 64))
        for (i, (x, y, w, h)) in enumerate(rects):
            img = cv2.imread(image)
            try:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            except Exception as e:
                print(str(e))

            cat = img[(y - 100):(y+h+100), (x-100):(x+w+100)].astype('float32')
            if np.size(cat) > 0:
                try:
                    cat = imutils.resize(cat, width=64)
                    #print(cat.shape)
                    #print(np.size(cat))
                    cv2.imwrite(path_aug + '/cropped_' + np.str(j) + ".jpg", cat)
                    j = j + 1
                    print(j)
                except Exception as e:
                    print(str(e))



print("\nCat faces found: " + np.str(j))



#You could then access an image by indexing it i.e.
