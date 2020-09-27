# Allstate Claims Severity- Kaggle Challange

## Introduction
This project is to develop automated methods of predicting the cost, and hence severity, of claims. And this is proposed by AllState, an insurance company.
The mission of the project is ultimately to be able to find more severe claims from many claims reports. So the claims adjuster can help those people who needs
immediate help will be able to be paid. <br> <br>
"When you’ve been devastated by a serious car accident, your focus is on the things that matter the most: family, friends, and other loved ones. Pushing paper with your insurance agent is the last place you want your time or mental energy spent. This is why Allstate, a personal insurer in the United States, is continually seeking fresh ideas to improve their claims service for the over 16 million households they protect."

## Data Source
I got this data source from Kaggle.<br> https://www.kaggle.com/c/allstate-claims-severity/   <br>
Each row in this dataset represents an insurance claim. You must predict the value for the 'loss' column. Variables prefaced with 'cat' are categorical, while those prefaced with 'cont' are continuous. <br>
There is no data dictionary for this dataset.

## Technology I use
- Python 3.7.3
- Pandas
- scipy
- EDA
  - matplotlib
  - seaborn
- Modelling
  - statsmodels - Linear Regression
  - Scikit Learn - Random Forest Regressor, Gradient Boosting Regressor, Multi-Layer Perception Regressor
  - skopt - hyperparameter tuning 


## Data Cleansing - missing value handling
train.csv has 188319 rows and 132 columns of data. I noticed that there are missing values in the dataset as below. <br>
{'id': 0, 'cat1': 0, 'cat2': 0, 'cat3': 0, 'cat4': 0, 'cat5': 0, 'cat6': 0, 'cat7': 0, 'cat8': 0, 'cat9': 0, 'cat10': 0, 'cat11': 0, 'cat12': 1, 'cat13': 1, 'cat14': 1, 'cat15': 1, 'cat16': 1, 'cat17': 1, 'cat18': 1, 'cat19': 1, 'cat20': 1, 'cat21': 1, 'cat22': 1, 'cat23': 1, 'cat24': 1, 'cat25': 1, 'cat26': 1, 'cat27': 1, 'cat28': 1, 'cat29': 1, 'cat30': 1, 'cat31': 1, 'cat32': 1, 'cat33': 1, 'cat34': 1, 'cat35': 1, 'cat36': 1, 'cat37': 1, 'cat38': 1, 'cat39': 1, 'cat40': 1, 'cat41': 1, 'cat42': 1, 'cat43': 1, 'cat44': 1, 'cat45': 1, 'cat46': 1, 'cat47': 1, 'cat48': 1, 'cat49': 1, 'cat50': 1, 'cat51': 1, 'cat52': 1, 'cat53': 1, 'cat54': 1, 'cat55': 1, 'cat56': 1, 'cat57': 1, 'cat58': 1, 'cat59': 1, 'cat60': 1, 'cat61': 1, 'cat62': 1, 'cat63': 1, 'cat64': 1, 'cat65': 1, 'cat66': 1, 'cat67': 1, 'cat68': 1, 'cat69': 1, 'cat70': 1, 'cat71': 1, 'cat72': 1, 'cat73': 1, 'cat74': 1, 'cat75': 1, 'cat76': 1, 'cat77': 1, 'cat78': 1, 'cat79': 1, 'cat80': 1, 'cat81': 1, 'cat82': 1, 'cat83': 1, 'cat84': 1, 'cat85': 1, 'cat86': 1, 'cat87': 1, 'cat88': 1, 'cat89': 1, 'cat90': 1, 'cat91': 1, 'cat92': 1, 'cat93': 1, 'cat94': 1, 'cat95': 1, 'cat96': 1, 'cat97': 1, 'cat98': 1, 'cat99': 1, 'cat100': 1, 'cat101': 1, 'cat102': 1, 'cat103': 1, 'cat104': 1, 'cat105': 1, 'cat106': 1, 'cat107': 1, 'cat108': 1, 'cat109': 1, 'cat110': 1, 'cat111': 1, 'cat112': 1, 'cat113': 1, 'cat114': 1, 'cat115': 1, 'cat116': 1, 'cont1': 1, 'cont2': 1, 'cont3': 1, 'cont4': 1, 'cont5': 1, 'cont6': 1, 'cont7': 1, 'cont8': 1, 'cont9': 1, 'cont10': 1, 'cont11': 1, 'cont12': 1, 'cont13': 1, 'cont14': 1, 'loss': 1}
<br>
I decided to simply delete the missing value because I suspect probably only one row has missing values for many columns. After dropping N/A values from the dataset, the number of rows became 188318, which clearly shows that one row had many missing values. I cleaned the dataset on <b>'[clean_data.ipynb](https://github.com/akira-nkgw/claims_severity/blob/master/clean_data.ipynb)'</b> and produced a new csv file for clean dataset, <b>'train_clean.csv'</b>. 


## Exploratory data analysis (EDA)
- Please refer to the EDA file for more details. <b>[EDA_claims_severity.ipynb](https://github.com/akira-nkgw/claims_severity/blob/master/EDA_claims_severity.ipynb)</b>
- For this project, because all the variables have no meaning in their name, and each categorical variable are encrypted into single alphabets like (A, B, C, D.. and so on), I will not be able to dig up the meaning of the dataset much. Therefore, I decided the focus on this project's EDA to be more statistical tests. 
- I noticed that the distribution for the target variable 'loss' is not normally distributed and has exceptionally large outliers as you can see the graph below.

![loss_bar](https://github.com/akira-nkgw/claims_severity/blob/master/images/loss_bar.png)

- This is understandable for insurance loss to be distributed like this above. For most of the loss, the mean of loss is $3,037.34 and the maximum loss we can find from the dataset is $121,012.25. This is possibly due to the fact that more severe events does not happen and it costs a lot if it really happens like the event of own death or other's.
### Categorical variables
- Since 'loss' is not normally distributed, I decided to use Mann-Withney U-test for binary categorical variables (cat1 to cat72)
- For multivariate categorical variables (cat72 to cat116), I decided to use ANOVA test to figure out if the 'loss' values for each class would have impact or not. 
- I set 1% as a significance level for both tests.
- The result was all the categorical variables, except cat88, reject the null hypothesis that their classes does not affect on the amount of loss (see more details in the EDA file).

### Numerical variables
- I chekced any continous variables if it is correlated to amount of 'loss'.
- When you look at the correlation table for loss, cont2, cont3, cont7 are more corelated which is at more than 0.10 correlation rate.
![corr](https://github.com/akira-nkgw/claims_severity/blob/master/images/corr.png)

## Model Building
- Please refer to the Model bulding file for more details <b>[models_claims_severe.ipynb](https://github.com/akira-nkgw/claims_severity/blob/master/models_claims_severe.ipynb)</b>
- Dataset ratio: 80% training, 20% testing
- 150,654 records for training, 37,663 records for testing
- I used Linear Regression, Random Forest, Gradient Boosting, and Neural Networks to predict the loss.

### Linear Regression Model (OLS)
- I used it as a bench mark score for the entire modeling. 
- The score was 1304.912
- It is not a bad score considering Linear Regression is not a complicated model unlike other ML models.
- Linear model is fitting to the training model well and you can see from the Adjusted R-squared:	0.528. 
- You can see that all three continuous variables are statistically significant at the 1% level and all of them have possitive impact on the loss.
  - cont2: the loss value is predicted to increases by $1353.13 for every one unit increase in cont2
  - cont3: the loss value is predicted to increases by $617.42 for every one unit increase in cont3
  - cont7: the loss value is predicted to increases by $684.00 for every one unit increase in cont7


### Random Forest Regressor
- The random forest is a model made up of many decision trees.
  - Random sampling of training data points when building trees
  - Random subsets of features considered when splitting nodes
  - The Random Forest is to not rely on any one individual tree, but pool the votes of each tree.
- The best score is 1244.31
- Important features are in the order of cat80, cont7, cont2, cat79, cat57.

![rf_importance](https://github.com/akira-nkgw/claims_severity/blob/master/images/rf_importance.png)


### Gradient Boosting Regressor
- I used the hyperparameter estimation for Gradient Boosting algorithm since this algorithm can be difficult to tune the model to not to be overfit. 
- The best score is 1214.5313
- Important features I've got from the GB model are cat80, cat79, cont7, cont2, cat12 as in the image below.

![gb_importance](https://github.com/akira-nkgw/claims_severity/blob/master/images/rf_importance.png)

### Neural Networks (Multi-layer Perception Regressor)
- The best score is 1193.08
- Contains only one layer with 80 neurons
- When extra neurons or layers are added, the mean absolute error gets bigger because of overfitting.
- The NN model with the best score has hyperparameters as below.

![nn_hyperparameter](https://github.com/akira-nkgw/claims_severity/blob/master/images/nn_hyperparameter.png)

## Summary
- Neural Networks worked the best among other ML algorithms for this problem and dataset.
- The most accurate Neural Networks model would have 1193.08 as the mean absolute error. This result means that the model's prediction would have error of $1193.08 dollars for each claim on average. 
- It would be interesting to see if there is casal relationship on some variables that were important features for Random Forest, Gradient Boosting and Linear Regression such as
  - continuous variables: cont2 and cont7  
  - categorical variables: cat12, cat57, cat79, cat80
- Those variables may have strong critical point for the insured to have higher loss in their accident.
- For the futher improvement for prediction, it would be nice to have a data dictionary for the dataset. If I have a data dictionary, it would be easier to understand the problem and feature engineering will be feasible to do as well.  

Feel free to contact me if you have any questions! <br>
Akira
