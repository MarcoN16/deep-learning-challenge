# deep-learning-challenge

This project aimed to help a nonprofit foundation Alphabet Soup to develop a tool that can help to select the applicants for funding with the best chance of success in their ventures. Employing machine learning techniques and neural networks, I analyzed the features present in the provided dataset to construct a binary classifier capable of determining whether applicants are likely to succeed if funded by Alphabet Soup.

The dataset consists of more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset, various columns capture metadata about each organization, including:
-	EIN and NAME—Identification columns
-	APPLICATION_TYPE—Alphabet Soup application type
-	AFFILIATION—Affiliated sector of industry
-	CLASSIFICATION—Government organization classification
-	USE_CASE—Use case for funding
-	ORGANIZATION—Organization type
-	STATUS—Active status
-	INCOME_AMT—Income classification
-	SPECIAL_CONSIDERATIONS—Special considerations for application
-	ASK_AMT—Funding amount requested
-	IS_SUCCESSFUL—Was the money used effectively

## Step 1: Preprocess the Data
processed the dataset using Pandas and scikit-learn's StandardScaler(). Initially, I removed less important columns like 'EIN' and 'Name' and standardized the data. Then, for two classes—'Application type' and 'Classification'—I calculated the number of data points for each unique value to determine a cutoff point for binning categorical variables into a new value 'Other'. For all other categorical values, I used one-hot encoding to convert them into binary variables. Subsequently, I split the data into training and testing sets using the 'train_test_split' function. The target variable for my model was the 'IS_SUCCESSFUL' column, while all other columns were utilized as features to train the model. These preprocessing steps set the stage for phase 2, where I compiled, trained, and evaluated the neural network model.


## Step 2: Compile, Train, and Evaluate the Model
I began by designing a basic neural network using TensorFlow to develop a binary classification model. This model aimed to predict whether an organization funded by Alphabet Soup would be successful, leveraging the dataset's features. Subsequently, I compiled, trained, and evaluated the binary classification model to assess its performance, calculating both the model's loss and accuracy. The obtained accuracy was 0.73, with a loss of 0.56, indicating subpar performance in selecting successful campaigns. To improve the model's efficacy, an optimization process will be necessary. Finally, I saved and exported the model to an HDF5 file named "AlphabetSoupCharity.h5.

## Step 3: Optimize the Model
Using TensorFlow, the objective was to optimize the model to achieve a target predictive accuracy higher than 75%.

### Second model
In my initial attempt to optimize the model, I decided to incorporate additional data into the training process, specifically including the company names to observe how the model would utilize this information. I used an automated process to determine the optimal number of units and hidden layers in my neural network using 'keras_tuner'. The final model consisted of 6 layers, including the input and output layers, with 71 neurons in the first layer, 16 in the second, 6 in the third, 16 in the fourth, and 1 in the fifth layer. While the model's performance showed improvement compared to before, achieving an accuracy of 0.79, the loss remained notably high at 0.50, suggesting that the model's predictions were still largely arbitrary. Additionally, there was a significant disparity between the model's accuracy and loss on the training data (accuracy_training = 0.97; loss = 0.08), indicating a potential issue with overfitting.  
As a final step, I saved and exported the optimized model to an HDF5 file named "AlphabetSoupCharity_Optimization.h5".

![output_accuracy_optimization1](https://github.com/MarcoN16/deep-learning-challenge/assets/150491559/310489d7-66b7-4a59-b9e3-b7ddb7a86222)
![output_loss_optimization1](https://github.com/MarcoN16/deep-learning-challenge/assets/150491559/2a039da4-1055-468d-8e28-cd2bca8c8a56)


### Third model
In my second attempt to optimize the model, I aimed to enhance its performance by including additional data. I transformed the 'INCOME_AMT' column into integers by incorporating the minimum and maximum income values. Utilizing the same automation script to determine the optimal number of units and layers, I analyzed the resulting best model. The final model comprised 6 layers, encompassing the input and output layers, with 31 neurons in the first layer, 16 in the second, 26 in the third, 1 in the fourth, 11 in the fifth, and 1 in the sixth layer. However, the model's performance exhibited a decrease compared to the previous attempt, achieving an accuracy of 0.74, while the loss remained notably high at 0.56. Additionally, the model's accuracy and loss on the training data were also suboptimal (accuracy_training = 0.74; loss = 0.54). I saved and exported the optimized model to an HDF5 file named "AlphabetSoupCharity_Optimization2.h5".

![output_loss_optimization2](https://github.com/MarcoN16/deep-learning-challenge/assets/150491559/a5a380ec-031d-4a1e-8873-f2df9958451a)
![output_accuracy_optimization2](https://github.com/MarcoN16/deep-learning-challenge/assets/150491559/743354f7-bd8e-4b35-8788-c7e2e1b5fab8)



### Fourth model
In my third attempt to optimize the model, I aimed to enhance its performance by reducing the amount of data used for training. Specifically, I removed multiple columns with unbalanced data, such as 'STATUS' and 'SPECIAL_CONSIDERATION', where a single value accounted for more than 90% of the occurrences. Additionally, instead of having separate columns for 'INCOME_AMT', I calculated the mean and used it as a feature. After scaling the data, I identified and eliminated all rows containing values more than 3 standard deviations from the mean in each column, resulting in a total of 21,972 data points. I then applied PCA analysis to reduce the dimensionality of the dataset (32 columns), and the first 7 principal components explained 100% of the variance in the data. Utilizing the same automation script to determine the optimal number of units and layers, I analyzed the resulting best model. The final model consisted of 4 layers, including the input and output layers, with 1 neuron in the first layer, 6 in the second, 16 in the third, and 1 in the fourth layer. However, the model's performance exhibited the lowest performance compared to the previous attempt, achieving an accuracy of 0.70, while the loss remained notably high at 0.60. This suggests that significant information were lost during the data cleaning process.

![outliers](https://github.com/MarcoN16/deep-learning-challenge/assets/150491559/cb37f610-0929-447b-9fd6-2dac3aad2f88)


### Final model
In the final attempt to optimize my model, I removed several columns with unbalanced data, such as 'STATUS' and 'SPECIAL_CONSIDERATION', and calculated the mean for the 'INCOME_AMT' column. Additionally, I assessed the number of data points for each unique value in the 'NAME' column to establish a cutoff point (<5) for binning categorical variables into a new value 'Other'. Using the same automation script to determine the optimal number of units and layers, I analyzed the resulting best model. The final model consisted of 5 layers, including the input and output layers, with 60 neurons in the first layer, 11 in the second, 11 in the third, 21 in the fourth, and 1 in the fifth layer. The final model exhibited the highest performance compared to the previous attempts, achieving an accuracy of 0.79, with the loss at 0.45. Similarly, the training data showed consistent results (accuracy_training = 0.81; loss = 0.41). This model met the target performance and demonstrated greater consistency across the training and testing datasets. As a concluding step, I saved and exported the optimized model to an HDF5 file named "AlphabetSoupCharity_Optimization4.h5".

![output_loss_optimization4](https://github.com/MarcoN16/deep-learning-challenge/assets/150491559/489cacd7-3616-42fe-ab1d-16c62730fb03)
![output_accuracy_optimization4](https://github.com/MarcoN16/deep-learning-challenge/assets/150491559/05869e18-86c0-4e76-87a8-fa7f8cc6a6a7)



### AdaBoostClassifier
For the final test to classify successful campaigns, I employed the AdaBoostClassifier algorithm, which is particularly advantageous when dealing with extensive datasets and aiming to construct a robust classification model capable of generalizing well to unseen data. This model exhibits effectiveness in handling noisy data and discerning complex relationships within the dataset. AdaBoostClassifier operates adaptively, adjusting its focus on challenging-to-classify instances by assigning higher weights to misclassified samples in each iteration. Furthermore, it leverages ensemble learning principles, combining multiple weak learners to form a potent classifier. Through the aggregation of predictions from these weak learners, AdaBoostClassifier enhances generalization and mitigates overfitting tendencies. The model's performance can be summarized as follows:
 confusion matrix:
-	True positive: 2689
-	False positive: 596
-	False negative: 1304
-	True negative: 3986
Calculation of precision, recall, and accuracy:
-	Precision to identify status = 0 (unsuccess) is 82% and status = 1(success) is 75%, indicating lower misclassification of false positives.
-	Recall to identify status = 0 (unsuccess) is 67% and status = 1 (success)  is 87%, indicating lower misclassification of false negatives.
The model appears to perform reasonably well overall, with a total accuracy of 78%. However, further analysis and consideration of specific use-case requirements may be necessary for a more comprehensive evaluation.

# References
IRS. Tax Exempt Organization Search Bulk Data Downloads. https://www.irs.gov/Links to an external site.


