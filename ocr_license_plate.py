from PyImageSearchANPR import PyImageSearchANPR
from imutils import paths
import argparse
import imutils
import cv2
from matplotlib import pyplot as plt

def cleanup_text(text):
    return "".join([c if ord(c) < 128 else "" for c in text]).strip()

# ap = argparse.ArgumentParser()
# # ap.add_argument("-i", "--input", required=True,
# #     help="path to input directory of images")
# ap.add_argument("-c", "--clear-border", type=int, default=-1,
#     help="whether or to clear border pixels before OCR'ing")
# ap.add_argument("-p", "--psm", type=int, default=7,
#     help="default PSM mode for OCR'ing license plates")
# ap.add_argument("-d", "--debug", type=int, default=-1,
#     help="whether or not to show additional visualizations")
# args = vars(ap.parse_args())

# initialize our ANPR class
anpr = PyImageSearchANPR()

# grab all image paths in the input directory
# imagePaths = sorted(list(paths.list_images(args["input"])))
def get_number(path):
    # imagePaths = sorted(list(paths.list_images(path)))
    num = plate_detection(path)
    return num

def display(im_path):
    dpi = 80
    im_data = plt.imread(im_path)
    height, width, depth = im_data.shape

    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis('off')
    # Display the image.
    ax.imshow(im_data, cmap='gray')
    plt.show()

def plate_detection(imagePaths):
    # loop over all image paths in the input directory

    # load the input image from disk and resize it
    # print(imagePath)
    # display(imagePath)
    image = cv2.imread(imagePaths)
    image = imutils.resize(image, width=600)

    # apply automatic license plate recognition
    (lpText, lpCnt) = anpr.find_and_ocr(image, psm=7,clearBorder=-1 > 0)
    print(lpText)
    # only continue if the license plate was successfully OCR'd
    if lpText is not None and lpCnt is not None:
        # fit a rotated bounding box to the license plate contour and
        # draw the bounding box on the license plate
        box = cv2.boxPoints(cv2.minAreaRect(lpCnt))
        box = box.astype("int")
        cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
        # compute a normal (unrotated) bounding box for the license
        # plate and then draw the OCR'd license plate text on the
        # image
        (x, y, w, h) = cv2.boundingRect(lpCnt)
        cv2.putText(image, cleanup_text(lpText), (x, y - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        # show the output ANPR image
        # print("[INFO] {}".format(lpText))
        # cv2.imshow("Output ANPR", image)
        # cv2.waitKey(0)
        
        return lpText
    
    else:
        print("Wrong aspect Ratio")
        return "Wrong aspect Ratio"

# path = input("Enter the path : ")
# get_number(path)