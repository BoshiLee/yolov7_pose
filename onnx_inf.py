import onnxruntime
import matplotlib.pyplot as plt
import torch
import cv2
from torchvision import transforms
import numpy as np
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts

if __name__ == '__main__':
    device = torch.device("cuda:0")

    image = cv2.imread('onnx_inference/img.png')
    image = letterbox(image, 960, stride=64, auto=True)[0]
    image_ = image.copy()
    image = transforms.ToTensor()(image)
    image = torch.tensor(np.array([image.numpy()]))

    print(image.shape)
    sess = onnxruntime.InferenceSession('yolov7-w6-pose-sim-yolo.onnx')
    out = sess.run(['output'], {'images': image.numpy()})[0]
    out = torch.from_numpy(out)

    output = non_max_suppression_kpt(out, 0.25, 0.65, nc=1, nkpt=17, kpt_label=True)
    output = output_to_keypoint(output)
    nimg = image[0].permute(1, 2, 0) * 255
    nimg = nimg.cpu().numpy().astype(np.uint8)
    nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
    for idx in range(output.shape[0]):
        plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)

    # matplotlib inline
    plt.figure(figsize=(8, 8))
    plt.axis('off')
    plt.imshow(nimg)
    plt.show()
    plt.savefig("tmp")