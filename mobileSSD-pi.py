from mvnc import mvncapi as mvnc
import numpy as np
import cv2
import time

# SSD Graph filepath
GRAPH = 'graph/graph'

# Set of detectable classes
CLASSES = ('background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

# Define required input image shape
input_size = (300, 300)

# Preprocess function to resize input image to required input shape
def preprocess(image):
    preprocessed = cv2.resize(image, input_size)
    preprocessed = preprocessed - 127.5
    preprocessed = preprocessed / 127.5
    return preprocessed.astype(np.float16)

# Randomize colors for output bounding boxes.
# `colors` is a (20,3) list containing random RGB values for 20 different classesself.
np.random.seed(3)
colors = 255 * np.random.rand(len(CLASSES), 3)

# Discover our NCS device
devices = mvnc.EnumerateDevices()
device = mvnc.Device(devices[0])
device.OpenDevice()

# Load graph from filesystem onto the NCS device
with open(GRAPH, 'rb') as f:
  graph_file = f.read()
graph = device.AllocateGraph(graph_file)

# Graph => load the image to it and  return a prediction
capture = cv2.VideoCapture(0)
_, image = capture.read()

# Grab height and width information from the image vector
height, width = image.shape[: 2]

# Begin Real-time video processing
while True:

    # Log start time
    start_time = time.time()

    # Grab image from video capture
    _, image = capture.read()

    # Preprocess image
    image_pro = preprocess(image)

    # Input image tensor into the graph
    graph.LoadTensor(image_pro, None)

    # Get the output result from the graph
    output, _ = graph.GetResult()

    # Get the number of valid boxes from the output
    num_valid_boxes = int(output[0])

    #`output` is a vector of (1+num_valid_boxes, 7)
    # Each row of the output except the first row corresponds to a valid bounding box and contains elements:
    #   [object-ness, classification, confidence, x1, y1, x2, y2]
    # We loop through each row of the output vector and set `i` to be the base index for each row
    for i in range(7, 7 * (1 + num_valid_boxes), 7):

        #Checks output between a range of elements in the output
        #If any of them are NaN, skip to the next iteration
        if not np.isfinite(sum(output[i + 1: i + 7])):
            continue

        #Get the prediction class label, the confidence, and classification index from the output vector
        classification = CLASSES[int(output[i + 1])]
        confidence = output[i + 2]
        color = colors[int(output[i + 1])]


        # Extract the image width and height and clip the boxes to the
        # image size in case boxes are outside of the image boundaries
        x1 = max(0, int(output[i + 3] * width))
        y1 = max(0, int(output[i + 4] * height))
        x2 = min(width, int(output[i + 5] * width))
        y2 = min(height, int(output[i + 6] * height))

        #Format label - Classification : Confidence
        label = '{}: {:.0f}%'.format(classification, confidence * 100)

        #Draw bounding box on image
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        #Draw text label on the image
        image = cv2.putText(image, label, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            color, 2)

    #Show image
    cv2.imshow('frame', image)

    #Console Log the Framerate for this iteration
    print('FPS = {:.1f}'.format(1 / (time.time() - start_time)))

    #If Escape key pressed, exit loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
device.CloseDevice()
