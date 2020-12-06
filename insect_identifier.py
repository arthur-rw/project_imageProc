from cropper import Cropper
from neural_net import Net
import torch
import cv2

resize_var = 100

device = torch.device('cpu')

with open('state.pt', 'rb') as f:
          net = torch.load('state.pt').to(device)
net.eval()

im = cv2.imread('c_gray.jpg')

im_cropped = Cropper.crop(im, [90, 90, 90])
# use below to test butterfly image
# im_cropped = cv2.imread('bm_1.jpg')

im_cropped = cv2.cvtColor(im_cropped, cv2.COLOR_BGR2GRAY)
im_cropped = cv2.resize(im_cropped, (resize_var, resize_var))

im_croppedTensor = torch.tensor(im_cropped)

im_croppedTensor = im_croppedTensor/255.0

im_croppedTensor_T = im_croppedTensor.view(-1, 1, resize_var, resize_var).to(device)

out = net(im_croppedTensor_T)
print(out)