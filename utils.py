import matplotlib.pyplot as plt
import cv2
import torch

def drawBoundingBoxes(image, detections, color = (0,0,255)):
    """Draw bounding boxes on an image.
    imageData: image data in numpy array format
    imageOutputPath: output image file path
    inferenceResults: inference results array off object (l,t,w,h)
    colorMap: Bounding box color candidates, list of RGB tuples.
    """
    img_with_dets = image.copy()
    min_conf = 0.9
    for det in detections:
      if det['confidence'] >= min_conf:
        x, y, width, height = det['box']
        label = det['label']

        cv2.rectangle(img_with_dets, (x,y), (x+width,y+height), (0,155,255), 2)
        imgHeight, imgWidth, _ = image.shape

        thick = int((imgHeight + imgWidth) // 900)
        cv2.putText(img_with_dets, label, (x, y - 12), 0, 1e-3 * imgHeight, color, thick//3)

    plt.figure(figsize = (5,5))
    plt.imshow(img_with_dets)
    plt.axis('off')

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_cuda(elements):
    if torch.cuda.is_available():
        return elements.cuda()
    return elements




