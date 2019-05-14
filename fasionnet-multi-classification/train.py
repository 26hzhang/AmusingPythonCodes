from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from .pyimagesearch.fashionnet import FashionNet
from imutils import paths
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import os

matplotlib.use("Agg")

# URL: https://www.pyimagesearch.com/2018/06/04/keras-multiple-outputs-and-multiple-losses/
# USAGE
# python train.py --dataset dataset --model output/fashion.model --categorybin output/category_lb.pickle
# --colorbin output/color_lb.pickle

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset (i.e., directory of images)")
ap.add_argument("-m", "--model", required=True, help="path to output model")
ap.add_argument("-l", "--categorybin", required=True, help="path to output category label binarizer")
ap.add_argument("-c", "--colorbin", required=True, help="path to output color label binarizer")
ap.add_argument("-p", "--plot", type=str, default="output", help="base filename for generated plots")
args = vars(ap.parse_args())

# initialize the number of epochs to train for, initial learning rate, batch size, and image dimensions
EPOCHS = 50
INIT_LR = 1e-3
BS = 32
IMAGE_DIMS = (96, 96, 3)

# grab the image paths and randomly shuffle them
print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

# initialize the data, clothing category labels (i.e., shirts, jeans,
# dresses, etc.) along with the color labels (i.e., red, blue, etc.)
data = []
categoryLabels = []
colorLabels = []

# loop over the input images
for imagePath in imagePaths:
    # load the image, pre-process it, and store it in the data list
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = img_to_array(image)
    data.append(image)
    # extract the clothing color and category from the path and update the respective lists
    (color, cat) = imagePath.split(os.path.sep)[-2].split("_")
    categoryLabels.append(cat)
    colorLabels.append(color)

# scale the raw pixel intensities to the range [0, 1] and convert to a NumPy array
data = np.array(data, dtype="float") / 255.0
print("[INFO] data matrix: {} images ({:.2f}MB)".format(len(imagePaths), data.nbytes / (1024 * 1000.0)))

# convert the label lists to NumPy arrays prior to binarization
categoryLabels = np.array(categoryLabels)
colorLabels = np.array(colorLabels)

# binarize both sets of labels
print("[INFO] binarizing labels...")
categoryLB = LabelBinarizer()
colorLB = LabelBinarizer()
categoryLabels = categoryLB.fit_transform(categoryLabels)
colorLabels = colorLB.fit_transform(colorLabels)

# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
split = train_test_split(data, categoryLabels, colorLabels, test_size=0.2, random_state=42)
(trainX, testX, trainCategoryY, testCategoryY, trainColorY, testColorY) = split

# initialize our FashionNet multi-output network
model = FashionNet.build(96, 96, num_categories=len(categoryLB.classes_), num_colors=len(colorLB.classes_),
                         final_activation="softmax")

# define two dictionaries: one that specifies the loss method for
# each output of the network along with a second dictionary that
# specifies the weight per loss
losses = {"category_output": "categorical_crossentropy", "color_output": "categorical_crossentropy"}
lossWeights = {"category_output": 1.0, "color_output": 1.0}

# initialize the optimizer and compile the model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(optimizer=opt, loss=losses, loss_weights=lossWeights, metrics=["accuracy"])

# train the network to perform multi-output classification
H = model.fit(trainX, {"category_output": trainCategoryY, "color_output": trainColorY}, epochs=EPOCHS, verbose=1,
              validation_data=(testX, {"category_output": testCategoryY, "color_output": testColorY}))

# save the model to disk
print("[INFO] serializing network...")
model.save(args["model"])

# save the category binarizer to disk
print("[INFO] serializing category label binarizer...")
f = open(args["categorybin"], "wb")
f.write(pickle.dumps(categoryLB))
f.close()

# save the color binarizer to disk
print("[INFO] serializing color label binarizer...")
f = open(args["colorbin"], "wb")
f.write(pickle.dumps(colorLB))
f.close()

# plot the total loss, category loss, and color loss
lossNames = ["loss", "category_output_loss", "color_output_loss"]
plt.style.use("ggplot")
(fig, ax) = plt.subplots(3, 1, figsize=(13, 13))

# loop over the loss names
for (i, name) in enumerate(lossNames):
    # plot the loss for both the training and validation data
    title = "Loss for {}".format(name) if name != "loss" else "Total loss"
    ax[i].set_title(title)
    ax[i].set_xlabel("Epoch #")
    ax[i].set_ylabel("Loss")
    ax[i].plot(np.arange(0, EPOCHS), H.history[name], label=name)
    ax[i].plot(np.arange(0, EPOCHS), H.history["val_" + name], label="val_" + name)
    ax[i].legend()

# save the losses figure
plt.tight_layout()
plt.savefig("{}_losses.png".format(args["plot"]))
plt.close()

# create a new figure for the accuracies
accuracyNames = ["category_output_acc", "color_output_acc"]
plt.style.use("ggplot")
(fig, ax) = plt.subplots(2, 1, figsize=(8, 8))

# loop over the accuracy names
for (i, name) in enumerate(accuracyNames):
    # plot the loss for both the training and validation data
    ax[i].set_title("Accuracy for {}".format(name))
    ax[i].set_xlabel("Epoch #")
    ax[i].set_ylabel("Accuracy")
    ax[i].plot(np.arange(0, EPOCHS), H.history[name], label=name)
    ax[i].plot(np.arange(0, EPOCHS), H.history["val_" + name], label="val_" + name)
    ax[i].legend()

# save the accuracies figure
plt.tight_layout()
plt.savefig("{}_accs.png".format(args["plot"]))
plt.close()
