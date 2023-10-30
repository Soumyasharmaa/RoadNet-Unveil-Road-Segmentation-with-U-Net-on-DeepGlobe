# RoadNet-Unveil-Road-Segmentation-with-U-Net-on-DeepGlobe
## About the Dataset

In disaster-prone regions, particularly in developing countries, the availability of maps and accessibility information is paramount for efficient crisis response. The DeepGlobe Road Extraction dataset addresses this need by presenting the task of automatically extracting road and street networks from satellite imagery.

### Data Overview

- **Training Data**: The dataset for the Road Challenge comprises 6,226 satellite images in RGB format, each with a size of 1024x1024 pixels.
- **Pixel Resolution**: These images feature a pixel resolution of 50cm, collected using DigitalGlobe's high-resolution satellite technology.

This dataset serves as a critical resource for training and evaluating road segmentation models, enabling the development of solutions that can aid disaster response efforts.

![Sample Image](https://github.com/Soumyasharmaa/RoadNet-Unveil-Road-Segmentation-with-U-Net-on-DeepGlobe/blob/main/2704_sat.jpg?raw=true)
*Sample satellite image from the DeepGlobe Road Extraction Challenge dataset.*

For more details about the dataset and how to access it, https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset.

## U-Net Model Architecture

The U-Net model architecture is designed for semantic segmentation tasks and is employed in this project to extract roads from satellite imagery. Below is a summary of the U-Net architecture:

```plaintext
__________________________________________________________________________________________________
 Layer (type)                Output Shape                 Param #   Connected to                  
==================================================================================================
 input_1 (InputLayer)        [(None, 128, 128, 3)]        0         []                            
                                                                                                  
 conv2d (Conv2D)             (None, 128, 128, 16)         448       ['input_1[0][0]']             
                                                                                                  
 dropout (Dropout)           (None, 128, 128, 16)         0         ['conv2d[0][0]']              
                                                                                                  
 conv2d_1 (Conv2D)           (None, 128, 128, 16)         2320      ['dropout[0][0]']             
                                                                                                  
 max_pooling2d (MaxPooling2  (None, 64, 64, 16)           0         ['conv2d_1[0][0]']            
 D)                                                                                               
                                                                                                  
 conv2d_2 (Conv2D)           (None, 64, 64, 32)           4640      ['max_pooling2d[0][0]']       
                                                                                                  
 dropout_1 (Dropout)         (None, 64, 64, 32)           0         ['conv2d_2[0][0]']            
                                                                                                  
 conv2d_3 (Conv2D)           (None, 64, 64, 32)           9248      ['dropout_1[0][0]']           
                                                                                                  
 max_pooling2d_1 (MaxPoolin  (None, 32, 32, 32)           0         ['conv2d_3[0][0]']            
 g2D)                                                                                             
                                                                                                  
 conv2d_4 (Conv2D)           (None, 32, 32, 64)           18496     ['max_pooling2d_1[0][0]']     
                                                                                                  
 dropout_2 (Dropout)         (None, 32, 32, 64)           0         ['conv2d_4[0][0]']            
                                                                                                  
 conv2d_5 (Conv2D)           (None, 32, 32, 64)           36928     ['dropout_2[0][0]']           
                                                                                                  
 max_pooling2d_2 (MaxPoolin  (None, 16, 16, 64)           0         ['conv2d_5[0][0]']            
 g2D)                                                                                             
                                                                                                  
 conv2d_6 (Conv2D)           (None, 16, 16, 128)          73856     ['max_pooling2d_2[0][0]']     
                                                                                                  
 dropout_3 (Dropout)         (None, 16, 16, 128)          0         ['conv2d_6[0][0]']            
                                                                                                  
 conv2d_7 (Conv2D)           (None, 16, 16, 128)          147584    ['dropout_3[0][0]']           
                                                                                                  
 max_pooling2d_3 (MaxPoolin  (None, 8, 8, 128)            0         ['conv2d_7[0][0]']            
 g2D)                                                                                             
                                                                                                  
 conv2d_8 (Conv2D)           (None, 8, 8, 256)            295168    ['max_pooling2d_3[0][0]']     
                                                                                                  
 dropout_4 (Dropout)         (None, 8, 8, 256)            0         ['conv2d_8[0][0]']            
                                                                                                  
 conv2d_9 (Conv2D)           (None, 8, 8, 256)            590080    ['dropout_4[0][0]']           
                                                                                                  
 conv2d_transpose (Conv2DTr  (None, 16, 16, 128)          131200    ['conv2d_9[0][0]']            
 anspose)                                                                                         
                                                                                                  
 concatenate (Concatenate)   (None, 16, 16, 256)          0         ['conv2d_transpose[0][0]',    
                                                                     'conv2d_7[0][0]']            
                                                                                                  
 conv2d_10 (Conv2D)          (None, 16, 16, 128)          295040    ['concatenate[0][0]']         
                                                                                                  
 dropout_5 (Dropout)         (None, 16, 16, 128)          0         ['conv2d_10[0][0]']           
                                                                                                  
 conv2d_11 (Conv2D)          (None, 16, 16, 128)          147584    ['dropout_5[0][0]']           
                                                                                                  
 conv2d_transpose_1 (Conv2D  (None, 32, 32, 64)           32832     ['conv2d_11[0][0]']           
 Transpose)                                                                                       
                                                                                                  
 concatenate_1 (Concatenate  (None, 32, 32, 128)          0         ['conv2d_transpose_1[0][0]',  
 )                                                                   'conv2d_5[0][0]']            
                                                                                                  
 conv2d_12 (Conv2D)          (None, 32, 32, 64)           73792     ['concatenate_1[0][0]']       
                                                                                                  
 dropout_6 (Dropout)         (None, 32, 32, 64)           0         ['conv2d_12[0][0]']           
                                                                                                  
 conv2d_13 (Conv2D)          (None, 32, 32, 64)           36928     ['dropout_6[0][0]']           
                                                                                                  
 conv2d_transpose_2 (Conv2D  (None, 64, 64, 32)           8224      ['conv2d_13[0][0]']           
 Transpose)                                                                                       
                                                                                                  
 concatenate_2 (Concatenate  (None, 64, 64, 64)           0         ['conv2d_transpose_2[0][0]',  
 )                                                                   'conv2d_3[0][0]']            
                                                                                                  
 conv2d_14 (Conv2D)          (None, 64, 64, 32)           18464     ['concatenate_2[0][0]']       
                                                                                                  
 dropout_7 (Dropout)         (None, 64, 64, 32)           0         ['conv2d_14[0][0]']           
                                                                                                  
 conv2d_15 (Conv2D)          (None, 64, 64, 32)           9248      ['dropout_7[0][0]']           
                                                                                                  
 conv2d_transpose_3 (Conv2D  (None, 128, 128, 16)         2064      ['conv2d_15[0][0]']           
 Transpose)                                                                                       
                                                                                                  
 concatenate_3 (Concatenate  (None, 128, 128, 32)         0         ['conv2d_transpose_3[0][0]',  
 )                                                                   'conv2d_1[0][0]']            
                                                                                                  
 conv2d_16 (Conv2D)          (None, 128, 128, 16)         4624      ['concatenate_3[0][0]']       
                                                                                                  
 dropout_8 (Dropout)         (None, 128, 128, 16)         0         ['conv2d_16[0][0]']           
                                                                                                  
 conv2d_17 (Conv2D)          (None, 128, 128, 16)         2320      ['dropout_8[0][0]']           
                                                                                                  
 conv2d_18 (Conv2D)          (None, 128, 128, 1)          17        ['conv2d_17[0][0]']           
                                                                                                  
==================================================================================================
Total params: 1941105 (7.40 MB)
Trainable params: 1941105 (7.40 MB)
Non-trainable params: 0 (0.00 Byte)
```
## Model Metrics

After training the U-Net model, the following metrics were obtained:

- Loss: 0.0905
- Dice Coefficient: 0.4025
- Intersection over Union (IoU): 0.2531
- Recall: 0.2685
- Precision: 0.8949

## Model Predictions

Here are some sample predictions made by the U-Net model on the satellite imagery:

![Prediction 1](https://github.com/Soumyasharmaa/RoadNet-Unveil-Road-Segmentation-with-U-Net-on-DeepGlobe/blob/main/Sample%201.png?raw=true)
![Prediction 2](https://github.com/Soumyasharmaa/RoadNet-Unveil-Road-Segmentation-with-U-Net-on-DeepGlobe/blob/main/Sample%202.png?raw=true)
![Prediction 3](https://github.com/Soumyasharmaa/RoadNet-Unveil-Road-Segmentation-with-U-Net-on-DeepGlobe/blob/main/Sample%203.png?raw=true)
![Prediction 4](https://github.com/Soumyasharmaa/RoadNet-Unveil-Road-Segmentation-with-U-Net-on-DeepGlobe/blob/main/Sample%204.png?raw=true)
![Prediction 5](https://github.com/Soumyasharmaa/RoadNet-Unveil-Road-Segmentation-with-U-Net-on-DeepGlobe/blob/main/Sample%205.png?raw=true)
![Prediction 6](https://github.com/Soumyasharmaa/RoadNet-Unveil-Road-Segmentation-with-U-Net-on-DeepGlobe/blob/main/Sample%206.png?raw=true)
![Prediction 7](https://github.com/Soumyasharmaa/RoadNet-Unveil-Road-Segmentation-with-U-Net-on-DeepGlobe/blob/main/Sample%207.png?raw=true)
![Prediction 8](https://github.com/Soumyasharmaa/RoadNet-Unveil-Road-Segmentation-with-U-Net-on-DeepGlobe/blob/main/Sample%208.png?raw=true)
![Prediction 9](https://github.com/Soumyasharmaa/RoadNet-Unveil-Road-Segmentation-with-U-Net-on-DeepGlobe/blob/main/Sample%209.png?raw=true)
![Prediction 10](https://github.com/Soumyasharmaa/RoadNet-Unveil-Road-Segmentation-with-U-Net-on-DeepGlobe/blob/main/Sample%2010.png?raw=true)







