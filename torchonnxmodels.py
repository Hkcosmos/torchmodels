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

with os.fdopen(os.open('ImagenetLabels.txt',os.O_RDONLY)) as f:
    class_labels = eval(f.read())
image = cv2.imread("fork.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (224, 224))
tensor = transforms.ToTensor()(image)
tensor = tensor.unsqueeze(0)
input_value = tensor.numpy()

#Resnet152
session0 = ort.InferenceSession("Resnet152.onnx")
input_name0 = session0.get_inputs()[0].name
output_name0 = session0.get_outputs()[0].name
start_time0 = time.time()
output0 = session0.run([output_name0], {input_name0: input_value})
end_time0 = time.time()
time0 = end_time0 - start_time0
output_onnnx0 = np.array(output0)
class_index0 = np.argmax(output_onnnx0)
predicted_class_onnx0 = class_labels[class_index0]
print("Predicted class in Resnet152 ONNX:", predicted_class_onnx0)
print("Time taken to predict the image in Resnet152 ONNX is:",time0)

#Inceptionv3
session1 = ort.InferenceSession("inception_v3.onnx")
input_name1 = session1.get_inputs()[0].name
output_name1 = session1.get_outputs()[0].name
start_time1 = time.time()
output1 = session1.run([output_name1], {input_name1: input_value})
end_time1 = time.time()
time_onnx1 = end_time1 - start_time1
output_onnnx1 = np.array(output1)
class_index1 = np.argmax(output_onnnx1)
predicted_class_onnx1 = class_labels[class_index1]
print("Predicted class in Inception_v3 ONNX:", predicted_class_onnx1)
print("Time taken to predict the image in Inception_v3 ONNX is:",time_onnx1)

#Efficientb1
session3 = ort.InferenceSession("Efficientb1.onnx")
input_name3 = session3.get_inputs()[0].name
output_name3 = session3.get_outputs()[0].name
start_time3 = time.time()
output3 = session3.run([output_name3], {input_name3: input_value})
end_time3 = time.time()
time_onnx3 = end_time3 - start_time3
output_onnnx3 = np.array(output3)
class_index3 = np.argmax(output_onnnx3)
predicted_class_onnx3 = class_labels[class_index3]
print("Predicted class in Efficientb1 ONNX:", predicted_class_onnx3)
print("Time taken to predict the image in Efficientb1 ONNX is:",time_onnx3)

#Mobilenetv2
session4 = ort.InferenceSession("mobilenet_v2.onnx")
input_name4 = session4.get_inputs()[0].name
output_name4 = session4.get_outputs()[0].name
start_time4= time.time()
output4 = session4.run([output_name4], {input_name4: input_value})
end_time4 = time.time()
time_onnx4 = end_time4 - start_time4
output_onnnx4 = np.array(output4)
class_index4 = np.argmax(output_onnnx4)
predicted_class_onnx4 = class_labels[class_index4]
print("Predicted class in Mobilenetv2 ONNX:", predicted_class_onnx4)
print("Time taken to predict the image in Mobilenetv2 ONNX is:",time_onnx4)

#Shufflenetv2x05
session5 = ort.InferenceSession("shufflenet_v2_x0_5.onnx")
input_name5 = session5.get_inputs()[0].name
output_name5 = session5.get_outputs()[0].name
start_time5 = time.time()
output5 = session5.run([output_name5], {input_name5: input_value})
end_time5 = time.time()
time_onnx5 = end_time5 - start_time5
output_onnnx5 = np.array(output5)
class_index5 = np.argmax(output_onnnx5)
predicted_class_onnx5 = class_labels[class_index5]
print("Predicted class in Shufflenetv2x05 ONNX:", predicted_class_onnx5)
print("Time taken to predict the image in Shufflenetv2x05 is:",time_onnx5)

#Shufflenetv2x10
session6 = ort.InferenceSession("shufflenet_v2_x1_0.onnx")
input_name6 = session6.get_inputs()[0].name
output_name6 = session6.get_outputs()[0].name
start_time6 = time.time()
output6 = session6.run([output_name6], {input_name6: input_value})
end_time6 = time.time()
time_onnx6 = end_time6 - start_time6
output_onnnx6 = np.array(output6)
class_index6 = np.argmax(output_onnnx6)
predicted_class_onnx6 = class_labels[class_index6]
print("Predicted class in Shufflenetv2x10 ONNX:", predicted_class_onnx6)
print("Time taken to predict the image in Shufflenetv2x10 is:",time_onnx6)


#Squeezenet10
session7 = ort.InferenceSession("squeezenet10.onnx")
input_name7 = session7.get_inputs()[0].name
output_name7 = session7.get_outputs()[0].name
start_time7 = time.time()
output7 = session7.run([output_name7], {input_name7: input_value})
end_time7 = time.time()
time_onnx7 = end_time7 - start_time7
output_onnnx7 = np.array(output7)
class_index7 = np.argmax(output_onnnx7)
predicted_class_onnx7 = class_labels[class_index7]
print("Predicted class in Shufflenetv2x05 ONNX:", predicted_class_onnx7)
print("Time taken to predict the image in Shufflenetv2x05 is:",time_onnx7)