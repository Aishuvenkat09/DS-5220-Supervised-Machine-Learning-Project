# DS-5220- Diabetic Retinopathy Detection

## Problem Description :
Diabetic retinopathy is the leading cause of blindness in the working-age population of the
developed world and is estimated to affect over 93 million people. Currently, detecting DR is a timeconsuming and manual process that requires a trained clinician to examine and evaluate photographs
of the retina. This poses a challenge in areas that traditionally lack access to skilled clinical facilities.
Moreover, the manual nature of DR screening methods promotes widespread inconsistency among
readers.

With color fundus photography as input, the goal of the project is to introduce an automatic DR
grading system capable of classifying images based on disease pathologies from four severity levels.
Throughout this study, we aim to elucidate a more effective means of classifying early-stage diabetic
retinopathy for potential clinical benefits.

## Dataset:
Diabetic retinopathy images were acquired from a Kaggle dataset of 35,000 images of size
84GB (California Healthcare Foundation 2015)6. The dataset comprises high-resolution retina images
taken under a variety of imaging conditions that vary in height and width between the low hundreds to
low thousands of pixels.

The left and right fields are provided for every subject. Images are labeled with a subject id as
well as either left or right (e.g. 1_left.jpeg is the left eye of patient id 1). A clinician has rated the
presence of diabetic retinopathy in each image on a scale of 0 to 4, according to the following scale:
 
 0 - No DR 1 - Mild 2 - Moderate 3 - Severe 4 - Proliferative 

![image](https://user-images.githubusercontent.com/37929675/110735772-69b0d080-81f8-11eb-97ba-ab02acfacb1c.png)

There are a total of 5 train sets each containing around 8,000 images of size 8.5 GB. Loading
the data and training the models is done using the Google Colab cloud server.
We found that the classes were very imbalanced with 73% of data containing images with no
severity (class 0).

Disease Class 0 - 25,810 images <br>
Disease Class 1 - 2,443 images<br>
Disease Class 2 - 5,292 images<br>
Disease Class 3 - 873 images<br>
Disease Class 4 - 708 images<br>

![image](https://user-images.githubusercontent.com/37929675/110735875-a5e43100-81f8-11eb-854e-8a999e5f8428.png)

## Approach and Methodology: 
 
### Data Augmentation with 5 Level Disease Classification: 

Our plan was to start with all the 35K images with 5 classes and build models using Google Colab. We first resized all the images to 500 by 500 pixels and trained our models and used data augmentation for populating other classes. Data augmentation is a necessary step in this case due to the relatively small number of images in class 1 to 4. Without data augmentation, class 1-4 will have limited data. Class 0 images were not augmented due to the large sample size. The following are the methods used for image augmentation: 

1.	Random Rotation: Picking a random degree of rotation between 25% on the left and 25% on the right 
2.	Random Noise:  Adding random noise to the image 
3.	Horizontal Flip: Flipping the image array of pixels 
4.	Random Brightness: Adding brightness at various levels 
5.	Vertical Flip: Flipping the image array of pixels 
 
We augmented each class to have 25,668 images per class to combat class imbalance and still allow the use of all healthy eye images. This process involves the use of all data from class 1 to 4, and half of the data from class 0. A Convolutional Neural Network (CNN) was used for this preliminary testing of data augmentation. This model was chosen due to the recent success of CNN models from numerous work such as (Krizhevsky 2013)1 to classify images through training with Graphical Processing Units (GPUs). Google Collab was utilized for a GPU processor. 
When trained on CNN, the performance of class 0 and class 4 began to increase thus affecting other classes. For classification, accuracy was the utilized metric due to how the training and validation set had equal distribution between each class. The overall validation set accuracy 27%. 
As a comparison, 566 images for each class was used to train each label without augmentation. This number was chosen based upon all available training images available for class 4 after 71 images per class were designated as validation and for testing. The overall accuracy with 566 per class was 28%. Results comparing augmented and non-augmented data is below on the same CNN model. 
 
Preliminary CNN Classification Detailed Validation Set Results 

|Class |	Augmented Data Accuracy |	566 per Disease Classification |
|------|--------------------------|--------------------------------|
|0 	   |     47% 	                |   43%                          |
|1 	   |     16% 	                |14%                             |
|2   	 |     19% 	                |24%                             | 
|3 	   |     11% 	                | 23%                            |
|4 	   |     43% 	                |37%                             | 
 
These results were largely due to inconsistent dimensions of the images and class 1, 2, and 3 are very similar and there are no distinct patterns found. Due to these results, this was strong evidence to proceed to a binary classification between healthy eyes versus the severely sick eyes (class 4). However, the relatively low sample size of the CNN was also a large contributor to the poor accuracy of severity class 1 to 3. 

These results were largely due to inconsistent dimensions of the images and class 1, 2, and 3
are very similar and there are no distinct patterns found. Due to these results, this was strong
evidence to proceed to a binary classification between healthy eyes versus the severely sick eyes
(class 4). However, the relatively low sample size of the CNN was also a large contributor to the poor
accuracy of severity class 1 to 3.

## Final Data
After testing with data augmentation and classification, we came to two finalized datasets. The
5 class and binary tasks utilized the same validation and test sets.

## Models:
### 1) Basic machine learning models
### KNN Model:

Besides utilizing deep learning models, we also used more basic machine learning models.
This analysis began with A KNN model. We defined two methods to take an input image and
converted it to a feature vector. The first method simply took an input image and resizes it to a fixed
size, and then flattened the RGB pixel intensities into a single list of numbers. The second method
accepted an input image and constructed a color histogram to characterize the color distribution of
the image. (Rosebrock 2016)12 Both methods had roughly 30% validation accuracy on the 566 per
class sample with a K of 3. The model did not resolve when running on a local computer across the 
5,000 per class size. The 25K per class training set was not used due to the smaller 5,000 per class
size did not resolve.

### PCA + Logistic Regression:
After concerns of kNN of having a slow testing speed, we moved on to Principal Component
Analysis (PCA) to shrink the dimensions of the input images. After PCA reduced the dimensions of
the images, we perform classification across all 5 disease classes using a Logistic Regression model.
Once the validation set was transformed by the same PCA transformation, the set had an accuracy of
30%. These results were comparable to the use of kNN.
To achieve higher accuracy binary classification of healthy versus severe disease (Class 4)
was done. We achieved 54% validation accuracy, which is slightly better than random guessing. The
ROC AUC value for the validation set was 56%.


![image](https://user-images.githubusercontent.com/37929675/110736775-3a9b5e80-81fa-11eb-9734-878d56c11fd4.png)


Basic machine learning models are not sophisticated enough for image classification to achieve highly predictive models. 
 
### 2) Multi-layer Perceptron: 
The next architecture utilizes a Multi-Layer Perceptron (MLP). An MLP can be viewed as a logistic regression classifier where the input is first transformed using a learned non-linear transformation. Multi-Layer Perceptron was trained on both versions of five disease level classification and binary classification are reported. 
 
### Binary Classification  
MLP has a 59% test set accuracy when trained on 11K images using Binary Classification. (1,0) Images used are of size 500 * 500 * 3.  
 
### Architecture: 
Dense Layer with 256 Units, with ‘relu’ activation function 
Dense Layer with 110 Units, with ‘relu’ activation function 
Dense Layer with 64 Units, with ‘relu’ activation function Dropout Layer at 0.2 rate 
Dense Layer with 32 Units, with ‘relu’ activation function 
Dropout Layer at 0.3 rate 
Flattening 
Dense layer 1 unit, with ‘sigmoid’  activation function 

![image](https://user-images.githubusercontent.com/37929675/110736956-86e69e80-81fa-11eb-9e18-c35d9668f735.png)

  
No doubt we need to go deeper or need convolutions to extract more complex features and patterns in order to get the most out of our model. 
 
### 3) Inception V3 Fine Tuning: 
Recent successes for image classification tend to utilize transfer learning. For example, analysis on this ER dataset achieved the most accuracy classification models with transfer learning on a VGG-18 with an accuracy of 60% across all 5 classes despite the use of only 500 images for each class (Mader 2017)2. Transfer learning using ResNet50 in PyTorch resulted in a similar tier of accuracy of 75% using similar quantities of data (Ilovescience 2019)3. 
After experiments in a fully trained model, a more advanced CNN model was used through transfer learning. A pretrained model called Inception Version 3 (V3) was utilized. This model was created by Google in 2015 with training through ImageNet data. The CNN model architecture has some unique characteristics (Raj 2008)10. For example, the model utilizes Factorized Convolutions where single large filters are replaced with multiple smaller filters to cover the same area as the larger filter. In addition, Inception V3 utilizes parallel filters of different sizes in the same layers. This model was utilized due to compact size of 92 MB to speed training. The architecture of the pre-trained model is below (Tsang 2018)14: 

![image](https://user-images.githubusercontent.com/37929675/110736935-7c2c0980-81fa-11eb-9456-3044bba33022.png)

  
Prior to training, there was a download of pre-trained weights for convolutional layers. A single Global Average Pooling layer was used for the output of the pre-trained weights from ImageNet. This pooling was used due to it’s ability to reduce the entire depth and height of the input tensor into a single neuron (Cook 2017)12. This significant pooling is used to combat overfitting. A single fully connected layer with ReLU activation was trained in fine-tuning with 1024 neurons. Next a 20% dropout layer was used due to originally significant overfitting. The output layer involved a Sigmoid activation function for binary classification and a Softmax activation function for 5 disease level classification. 
 
5 Disease Level Classification 
Both the classification and binary models utilized a stochastic gradient descent algorithm to speed the training process. A learning rate of 0.0001 was utilized. The principle of momentum was utilized for the training to reduce the oscillations common in stochastic gradient descent. Momentum results in changes in the weights to more frequently follow the direction based on previous changes in the weights (Rumelhart 1986)11 Training of the classification model is below. Once training was complete, the plots of the training and validation accuracy as well as binary cross entropy loss showed relatively pronounced overfitting, even if some generalized understanding of the images was achieved over time. 
 	  
![image](https://user-images.githubusercontent.com/37929675/110736985-97971480-81fa-11eb-9dbe-ecc067682df3.png)
![image](https://user-images.githubusercontent.com/37929675/110737007-9cf45f00-81fa-11eb-8b2d-7029a3e890fe.png)

 
Validation set results for the 5 disease level classification was not very predictive, but not poor results with 5 classes. The overall validation set accuracy was 39%. Accuracy was used as the main metric due to how each class was evenly distributed in the training and validation set. Precision and recall for each of the 4 classes were very similar, so it is omitted here. 
 
Inception V3 Classification Accuracy Validation Set Results by Class 
Class 0 	Class 1 	Class 2 	Class 3 	Class 4 
55% 	14% 	10% 	53% 	62% 
 
As can be seen below, the recall for Class 0 was relatively high at 55%. In addition, class 3 and class 4 had relatively high accuracy at 53% and 62% respectively. The highest accuracy for class 0 and 4 gave addition evidence that a binary classification model between class 0 and 4 should be attempted. 
 
Binary Classification 
Binary classification between healthy and class 4 had high accuracy and relatively mild overfitting. Training involved 33 epochs to maximize validation set accuracy. 
 
Both classes had very similar accuracies, so there were no concerns that the model’s accuracy was due to overreliance on a single class. 
 
 ![image](https://user-images.githubusercontent.com/37929675/110737082-bbf2f100-81fa-11eb-8e5a-e4cc170c15ee.png)

 
Inception V3 - All Health and Severe Disease Images Used 
Prior to the end of the project, a final attempt was made to utilize more of the entire dataset. All 25,668 records for healthy images were used. All 566 records of disease class 4 were used, but the records were duplicated until there were 25,668 records. Due to each severe disease image being duplicated 500 times, overfitting was originally a significant problem until the neurons in the fully connected layer were reduced to 512 and dropout was increased to 50%. However, after 2 hours of training the validation set accuracy was only 82% after 8 epochs. This model was not overfitting, but it was not able to converge to a much higher accuracy. See results below. 
  
 ![image](https://user-images.githubusercontent.com/37929675/110737112-c90fe000-81fa-11eb-9560-3f0f02ca5dd0.png)
![image](https://user-images.githubusercontent.com/37929675/110737131-cd3bfd80-81fa-11eb-84c4-d3b96081fc0b.png)

 
The binary classification with 566 still performed much better with 87% validation set accuracy, so that the original dataset will be used for the conclusions. 

## Results & Conclusions: 

Our final conclusions are based off of a test set utilizing the same distribution as the validation set of 71 images for each of the 5 classes. 
 
### 5 Disease Level Classification 
Five image type classification also utilized 71 images for each test set class. 
 
Test Set 5 Disease Level Classification Accuracy Results 

|PCA + Logistic Regression| 	Multi-Level Perceptron |	Inception V3 Fine Tuning |
|----|--------|----|
|30% |	32% 	|37% |
 
Inception V3 Detailed Classification Test Set Results 

|Class |	Precision |	Recall |	Accuracy |
|------|------------|--------|-----------|
|0 |	34% |	55% |	55% |
|1 |	37% |	15% |	15% |
|2 |	30% |	18% |	18% |
|3 |	36% |	51% |	51% |
|4 |	52% |	46% |	46% |
 
The result in the validation set having higher accuracy for Inception V3 from class 4 versus class 3, and this being switched in the test set gives some evidence that the class 3 images are relatively distinct from class 0 images. The model likely would have been improved if it utilized binary classification to distinguish between healthy eyes versus class 4 images. In addition, all models do not result in very strong overall accuracies. This is likely contributed to the most successful data sets only involved 566 images for each class. Data sets with more data using data augmentation or resampling did not improve the performance of our models. 
 
### Binary Classification 

Binary classification comparing the healthy eyes (class 0) versus the most sick eyes (class 4) utilized 142 total images across the 2 disease classes. Since the training and test sets have an equal distribution between all classes, accuracy is a strong predictor. In addition, ROC AUC was utilized to confirm there are not large differences for binary classification, which there was not. 
 
Test Set Binary Classification Results 

|Model |	Test Binary Classification Accuracy |	Test Binary Classification ROC AUC |
|------|--------------------------------------|------------------------------------|
|PCA + Logistic Regression |	55% |	55% |
|Multi-Level Perceptron |	54% |	68% |
|Inception V3 Fine Tuning |	85% |	90% |
 
ROC Curves for Test Set Binary Classification 

 	      PCA + Logistic                      Multi-Level Perceptron                     Inception V3 
        
  ![image](https://user-images.githubusercontent.com/37929675/110737697-d9748a80-81fb-11eb-8de8-f120f1f7a796.png)

 
PCA and Logistic Classification had successes during five 
disease level classification, but the limited quantities of data made it difficult for the model to perform any better compared to random chance. PCA tends to utilize access to all data before reduction in dimensions, and this requires better hardware. The relatively poor results for even a powerful model such as Multi-Level Perceptron points to the need for more data to assist in training. In addition, more data could likely result in extremely high accuracy using transfer learning to improve upon the 85% accuracy and 90% ROC AUC in the test set. While the training data was relatively small, it resulted in a balanced CNN model. Accuracies in Inception V3 did not differ between healthy and severely sick eyes. Precision and recall also did not diverge from accuracy metrics by very much. 
 
## Future Steps 

There are multiple possible improvements for data pre-processing. Better method to crop image based on edge between black background and the retina may result in better data. The main ideas of pre-processing involve ways to use of more data. The best models were not the systems which had the most training data, but the ones with only 566 images for each disease level. However, image classification especially using deep learning for difficult tasks requires more data more in line with the number of trainable parameters. More data can be attempted to be used by depending upon weights for each class to combat class imbalance. Use of augmented data to train on 25K images per class for Binary Classification can also be compared with weights on each class. If use of more extensive data augmentation or resampling is done, downsizing the images to 256 by 256 can allow for analysis of more images on the same hardware. 

Use more powerful cloud computing to allow larger trained models, and the use of more data. For example, training the entire Inception V3 from scratch by starting with weights that were trained from ImageNet could result in more appropriate features for use of medical images. However, this would likely require the use of more sophisticated hardware, such as using Amazon Web Services. With more hardware resources, larger pre-trained models such as NASNetLarge can also be utilized. 
 
