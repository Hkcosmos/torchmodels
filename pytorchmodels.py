import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import EfficientNet_B1_Weights
import numpy as np
import onnxruntime as ort
import os
import cv2
import time
import numpy as np

image = cv2.imread('shuttle.jpg')
height, width = 224, 224  # example size
image = cv2.resize(image, (width, height))

# Convert BGR to RGB format
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert to PyTorch tensor
image = np.transpose(image, (2, 0, 1))  # swap channels to (C, H, W) format
image = torch.from_numpy(image).float()  # convert to float tensor
image /= 255.0 

with os.fdopen(os.open('ImagenetLabels.txt',os.O_RDONLY)) as f:
    class_labels = eval(f.read())

#Resnet152
model0 = models.resnet152(pretrained=True)
model0.eval()
starttime = time.time()
with torch.no_grad():
    output = model0(image.unsqueeze(0))
endtime = time.time()
time_torch0 = endtime - starttime
pred_idx0 = torch.argmax(output).item()
predicted_class_torch0 = class_labels[pred_idx0]

print("Predicted class in PYTORCH Resnet152: ", predicted_class_torch0)
print("Total time taken in predicting in Resnet152: ",time_torch0)

#Inception_v3
model1 = models.inception_v3(pretrained=True)
model1.eval()
starttime1 = time.time()
with torch.no_grad():
    output = model1(image.unsqueeze(0))
endtime1 = time.time()
time_torch1 = endtime1 - starttime1
pred_idx1 = torch.argmax(output).item()
predicted_class_torch1 = class_labels[pred_idx1]

print("Predicted class in PYTORCH Inception_v3: ", predicted_class_torch1)
print("Total time taken in predicting in inception_v3: ",time_torch1)

#Efficientb1
model2 = models.efficientnet_b1(pretrained=True)
model2.eval()
starttime2 = time.time()
with torch.no_grad():
    output = model2(image.unsqueeze(0))
endtime2 = time.time()
time_torch2 = endtime2 - starttime2
pred_idx2 = torch.argmax(output).item()
predicted_class_torch2 = class_labels[pred_idx2]

print("Predicted class in PYTORCH Efficientb1: ", predicted_class_torch2)
print("Total time taken in predicting in Efficientb1: ",time_torch2)

model3 = models.mobilenet_v2(pretrained=True)
model3.eval()
starttime3 = time.time()
with torch.no_grad():
    output = model3(image.unsqueeze(0))
endtime3 = time.time()
time_torch3 = endtime3 - starttime3
pred_idx3 = torch.argmax(output).item()
predicted_class_torch3 = class_labels[pred_idx3]

print("Predicted class in PYTORCH Mobilenetv2: ", predicted_class_torch3)
print("Total time taken in predicting in Mobilenetv2: ",time_torch3)

#Shufflenet0_5
model4 = models.shufflenet_v2_x0_5(pretrained=True)
model4.eval()
starttime4 = time.time()
with torch.no_grad():
    output = model4(image.unsqueeze(0))
endtime4 = time.time()
time_torch4 = endtime4 - starttime4
pred_idx4 = torch.argmax(output).item()
predicted_class_torch4 = class_labels[pred_idx4]

print("Predicted class in PYTORCH Shufflenet05: ", predicted_class_torch4)
print("Total time taken in predicting in Shufflenet05: ",time_torch4)

#Shufflenetv2x10
model5 = models.shufflenet_v2_x1_0(pretrained=True)
model5.eval()
starttime5 = time.time()
with torch.no_grad():
    output = model5(image.unsqueeze(0))
endtime5 = time.time()
time_torch5 = endtime5 - starttime5
pred_idx5 = torch.argmax(output).item()
predicted_class_torch5 = class_labels[pred_idx5]

print("Predicted class in PYTORCH Shufflenetv2x10: ", predicted_class_torch5)
print("Total time taken in predicting in Shufflenetv2x10: ",time_torch5)


#Squeezenet11
model6 = models.squeezenet1_0(pretrained=True)
model6.eval()
starttime6 = time.time()
with torch.no_grad():
    output = model6(image.unsqueeze(0))
endtime6 = time.time()
time_torch6 = endtime6 - starttime6
pred_idx6 = torch.argmax(output).item()
predicted_class_torch6 = class_labels[pred_idx6]

print("Predicted class in PYTORCH Squeezenet10: ", predicted_class_torch6)
print("Total time taken in predicting in Squeezenet10: ",time_torch6)