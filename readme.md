# Predict Clicked Ads Customer Classification by using Machine Learning
- - - 
**Tool** : Jupyter Notebook <br>
**Programming Language** : Python <br>
**Libraries** : Pandas, NumPy, Seaborn, Matplotlib, scikit -learn <br>
**Dataset** : Clicked Ads from Rakamin Academy <br>

**Table of Contents**
- [STAGE 0: Problem Statement](#stage-0-problem-statement)
    - [Background](#background)
    - [Goal](#goal)
    - [Objective](#objective)
- [STAGE 1: Data Preparation](#stage-1-data-preparation)
    - [Insight](#insight)
    - [Descriptive Analysis](#descriptive-analysis)
- [STAGE 2: Data Exploration](#stage-2-data-exploration)
    - [Univariate Analysis](#univariate-analysis)
    - [Bivariate Analysis](#bivariate-analysis)
    - [Multivariate Analysis](#multivariate-analysis)
- [STAGE 3: Preprocessing](#stage-3-data-preprocessing)
- [STAGE 4: Modeling And Evaluation](#stage-4-modeling-and-evaluation)
    - [Confusion Matrix](#confusion-matrix)
    - [Learning Curve](#learning-curve)
    - [Business Insight](#business-insight)
- [STAGE 5: Business Recomendation And Simulation](#stage-5-business-recomendations-and-simulation)
    - [Conclusion](#conclusion)
<br>

## Stage 0 Problem Statement

## Background
As technology advances, businesses must be able to optimize their digital advertising tactics in order to attract new customers at the lowest possible cost. This is done to increase conversions, or the number of potential customers who make a purchase after clicking on an advertisement. However, in order to reach this goal, businesses must be able to estimate click-through rates accurately. A precise click-through rate is critical to the success of a digital advertising campaign. Companies may pay big expenses without achieving significant outcomes if appropriate predictions are not made.

## Problem
The business team needs to optimize their digital advertising strategies in order to encourage potential users to click on a product while keeping the costs to a minimum.

## Goal
Make target marketing effective by using machine learning so that it can increase the click-through rate (CTR) and reduce costs incurred.

## Objective
- With 90% accuracy, predict which users are likely to click on ads. 
- Gain insight into probable trends of consumers who click on ads.
- Based on the study and model results, make business recommendations.

## Business Metrics
- Click Through Rate (CTR)
- Total Cost <br>

## Data
The data to be used is ``Clicked Ads Dataset.csv.`` The data has nine features with one target, the following is the variable information used:  

Variable Information:

| Column   | Descriptioan |
|-----------|--------------|
|Daily Time Spent on Site| : Length of stay at a site (daily) in minutes|
|Age | : User's age in years|
|Area Income |: User income in rupiah units|
|Daily Internet Usage | : Daily internet usage in minutes|
|Male | : Gender user|
|Timestamp | : When a user visits a site|
|Clicked on Ad | : Clicking on ads or not|
|city | : City of origin of the user|
|province | : Province of origin of the user| 

# STAGE 1 Data Preparation
- Handling missing values
- Handling Duplicated Data
- Check the type and consistency of values
- Checking for outliers or unusual data (anomalies)

### Insight 
- Overall this dataset has 1000 rows and 11 variables
- Missing values found were only around 0.4% in the `income`, `daily time spent`, `daily internet usage`, `male` columns
- no duplicated data
- an incorrect data type was found in the `timestamp` column which should have a datetime
- all numerical features do not seem to have outliers, so we will validate further later
- the male column should be replaced with gender

### Descriptive analysis
- some columns such as `age`, `income` have a distribution that is close to normal, then columns such as `daily time spent`, `daily internet usage` have a bimodal tendency
- The average user spends `daily time spent on site` which is 65 minutes with a minimum of 33 minutes / day
- for user ages ranging from 19 - 61 years, with the average user who surfs the web most often being 36 years
- The average user only uses the internet for 180 minutes (3 hours) 
- For The Details please check in notebook.

# STAGE 2. Data Exploration 

### Univariate analysis
<p align="center">
    <kbd><img width="700" alt="Distplot" src="https://github.com/fauzanheryka/Predict-Clicked-Ads-Customer-Classification-by-using-Machine-Learning/assets/141212116/e206f178-e601-49a6-84ca-bd42f7fbee33"></kbd><br>
    Figure 1 - Displot Skewness
</p>

### Insight : 
It has been confirmed that the distribution of numerical features is bimodal and almost normal, soo let's validate outlier.

<p align="center">
    <kbd><img width="800" alt="Boxplot" src="https://github.com/fauzanheryka/Predict-Clicked-Ads-Customer-Classification-by-using-Machine-Learning/assets/141212116/6f0cf580-29bb-4982-863d-c7480982730b"></kbd><br>
    Figure 2 - Boxplot
</p>

### Insight : 
It can be seen that there are no extreme outliers, in the income feature there are visible outliers but if you look at the QQ plot in DataPrep it is still a collective outlier.

<p align="center">
    <kbd><img width="800" alt="Boxplot" src="https://github.com/fauzanheryka/Predict-Clicked-Ads-Customer-Classification-by-using-Machine-Learning/assets/141212116/cfa13789-2488-49ed-a29e-f854f98243e4"></kbd><br>
    Figure 3 - Countplot
</p>

### Insight : 
- Here we can see that the target variable `clicked on ad` has a balanced distribution
- The categorical features look clean and there are no unknown values

<p align="center">
    <kbd><img width="400" alt="Boxplot" src="https://github.com/fauzanheryka/Predict-Clicked-Ads-Customer-Classification-by-using-Machine-Learning/assets/141212116/222b15a6-d206-457f-a3f9-2b0f26d02ded"></kbd><br>
    Figure 4 - Ratio Clicked Ads
</p>

### Insight : 
It is visible that the target feature has a balanced ratio, indicating that it is not unbalanced and that additional sampling is not required.

### Bivariate Analysis

<p align="center">
    <kbd><img width="800" alt="Boxplot" src="https://github.com/fauzanheryka/Predict-Clicked-Ads-Customer-Classification-by-using-Machine-Learning/assets/141212116/04ebde10-7c5b-4e4e-902c-a7170c8ddfda"></kbd><br>
    Figure 5 - Countplot
</p>

### Insight : 
- In terms of `gender` features, it can be seen that women click on advertisements more often compared to men who tend not to click on advertisements.
- It can be seen in the advertising category, it turns out that automotive clicks on advertisements more often than other categories.
- If you look more closely, the comparison between the number of people who clicked on the ad and those who didn't click on the ad doesn't have a significant difference.

<p align="center">
    <kbd><img width="800" alt="Boxplot" src="https://github.com/fauzanheryka/Predict-Clicked-Ads-Customer-Classification-by-using-Machine-Learning/assets/141212116/770d7db1-6cd7-4e78-83fc-0645b9f13761"></kbd><br>
    Figure 6 - HistPlot
</p>

### Insight  : 
- The best time for a customer to see an advertisement click is between 40 and 60 minutes, as this is when customers are most likely to click on it. 
- Customers who often use the internet tend not to click on ads, this is inversely proportional to customers who don't use the internet often, who often click on ads.
- Customers who click on advertisements are often between the ages of 35 and 50. This could be because younger people are more likely to be tech savvy and may be more aware of how to avoid advertisements by using ad blocking software or other techniques.

**Because we only have analysis based on minutes, we will try time analysis using months, weekdays from extracting the `timestamp` feature**

<p align="center">
    <kbd><img width="800" alt="Boxplot" src="https://github.com/fauzanheryka/Predict-Clicked-Ads-Customer-Classification-by-using-Machine-Learning/assets/141212116/6ffbccc7-bf93-44db-9b0b-f4acedfd7cde"></kbd><br>
    Figure 7 - Lineplot Month
</p>

## Insight : 
- Customers who clicked on adverts climbed up in February and May, after that it decreased.
- Customers who clicked on adverts reduced between March and June, which could be because particular businesses or items have peak seasons. If the marketed product or service is in high demand in February but drops in June, this could be a natural business trend.
- To overcome this, extensive analytical reviews of advertising data, listening to client input, and possibly adapting marketing techniques to match changes in customer wants or preferences are all needed.

<p align="center">
    <kbd><img width="800" alt="Boxplot" src="https://github.com/fauzanheryka/Predict-Clicked-Ads-Customer-Classification-by-using-Machine-Learning/assets/141212116/7399fa52-67ee-4bca-ad56-78c4c1fc4dac"></kbd><br>
    Figure 8 - Lineplot Days According to ISO 8601.
</p>

### Insight : 
- On Tuesdays, users may experience routine changes that influence when and how they interact with advertisements. It may be required to determine whether there is a shift in consumer behavior trends on that particular day.
- On the other hand, user clicks on ads increase greatly on Friday, possibly because it is a day when people are more inclined to relax and have more free time, especially if it is followed by a weekend. This increases the likelihood that consumers will interact with advertisements.

## Multivariate analysis 

<p align="center">
    <kbd><img width="800" alt="Boxplot" src="https://github.com/fauzanheryka/Predict-Clicked-Ads-Customer-Classification-by-using-Machine-Learning/assets/141212116/45704168-fdc1-4056-8d6a-a351e5237732"></kbd><br>
    Figure 9 - Pairplot
</p>

### Insight : 
The pair plot above shows that the target features are balanced, so it will produce a good model.

<p align="center">
    <kbd><img width="800" alt="Boxplot" src="https://github.com/fauzanheryka/Predict-Clicked-Ads-Customer-Classification-by-using-Machine-Learning/assets/141212116/f2a67005-1bba-4ef3-a3f1-fbb5fdcb30e2"></kbd><br>
    Figure 10 - Heatmap
</p>

### Insight : 
- `daily time spent on site` and `area income` have a fairly high correlation.
- `daily time spent on site` and `daily internet usage` also have a high correlation, but it is still not enough to be said to be redundant so a vif test will be carried out later.
- `area income` and `daily internet usage` also have a pretty good correlation too.
That's all the features that have a correlation with other features, but because the target feature has not yet been encoded, a heatmap will be carried out after the data has been preprocessed.<br>

# STAGE 3. Data PreProcessing 
This process includes handling missing values, selecting and transforming features, handling outliers, extracting features, and deciding which targets to utilize for modeling.

### Handle missing value
For missing value handles in the four columns : 

- `The Daily Time Spent on Site` column will be filled with median
- `The Area Income` column will be filled with the median due to skew  
- `The Daily Internet` Usage column will be filled with median  
- `The Gender` field will be filled with the mode  

Filling in missing values for numerical features uses the median value because the median is relatively robust against outliers (not affected by very high or low values) and does not affect the distribution of the data at the beginning.
for feature gender fill with mode.
## Handle outlier 
If the outliers are still normal/collective outliers, as mentioned above, and a qq plot analysis has been performed, no handling will be performed. 
## Feature Encoding 
One hot encoding : `category`
Label Encoding : `gender,weekdays,clicked on ads`

## Feature Selection 

<p align="center">
    <kbd><img width="800" alt="Boxplot" src="https://github.com/fauzanheryka/Predict-Clicked-Ads-Customer-Classification-by-using-Machine-Learning/assets/141212116/6c9be26a-f5be-4f8a-8f94-0b3aa46a9825"></kbd><br>
    Figure 11 - Heatmap
</p>

From the heatmap above features that have a strong correlation with the target feature:
- daily time spent
- age
- income area
- daily internet usage
- gender

because some of these features have indications of being redundant, we will carry out a multicollinearity test, and also to take the best features for modeling, we will use kbest selection.

<p align="center">
    <kbd><img width="300" alt="Boxplot" src="https://github.com/fauzanheryka/Predict-Clicked-Ads-Customer-Classification-by-using-Machine-Learning/assets/141212116/4c608357-5220-4b0a-80b4-a21bf3fc63b7"></kbd><br>
    Figure 12 - Vif
</p>
Selecting features with a VIF below <10 aims to optimize the regression model. By reducing the level of multicollinearity, the model can provide more stable and reliable results.

Following that, we separate using variables X and Y to apply kbest selection.
### Kbest Selection
Based on kbest 10 important features only : 
- `daily_time_spent_on_site`, 
- `age`, `area_income`, 
- `daily_internet_usage`,
- `gender`, 
- `category_Bank`,
- `category_Fashion`, 
- `category_Finance`, 
- `category_Furniture`,
- `category_Health`

# STAGE 4. Modeling And Evaluation 
Two different tests were conducted to make predictions on ad clicks. In the first experiment, the model is trained using the default train data. This experiment makes use of train data in its default or without any extra changes. Meanwhile, the data in the second trial was normalized using StandardScaler. Because the data distribution is close to normal, standardization is required to ensure that the data has a same scale. 

We focus on the evaluation measure `f1` score for this dataset since we want to maximize CTR by providing advertising to users who are actually interested and ensuring as many actual clicks as possible to lower the total cost used, therefore `f1` is the optimal evaluation metric. 

<p align="center">
    <kbd><img width="700" alt="Boxplot" src="https://github.com/fauzanheryka/Predict-Clicked-Ads-Customer-Classification-by-using-Machine-Learning/assets/141212116/a511604e-8a31-4964-9adf-eafb2a649705"></kbd><br>
    Figure 13 - model experiment(Without Scalling)
</p>

### Results :
- we can see that logistic regression without scaling has a very bad score, perhaps because regression is a linear model and the features without scaling have a lot of variance, so that is what causes the logistic score to be very bad
- For a model that is quite good, it is used in gradient boosting which has a reasonable F1 score and also if you look at Auc Roc, the model can be said to work well

<br>

<p align="center">
    <kbd><img width="700" alt="Boxplot" src="https://github.com/fauzanheryka/Predict-Clicked-Ads-Customer-Classification-by-using-Machine-Learning/assets/141212116/bb202543-93f4-4986-a2ad-e7b66db86980"></kbd><br>
    Figure 14 - model experiment(With Scalling)
</p>

### Result : 
The model's purpose is to anticipate how many potential consumers will click on an advertisement. As a result, we must reduce the frequency of False Positives, or the incorrect prediction that consumers who do not click on advertisements will do so in the future. The best strategy to balance recall and precision is to use F1 score rather than retargeting the wrong market, which could result in losses because we wasted money on advertising on the wrong target.

Upon scaling the features, it is obvious that `Logistic Regression` and `KNN` have the highest f1 score and stable auc roc, indicating that the model is potentially effective. To ascertain which model is the best, additional analysis will be conducted by examining the confusion matrix, learning curve.

## Hyperparameters Tuning 
There is no difference between before hyper and after hyper, so our assumption is that the model has reached the maximum score.

## Confusion Matrix 
<p align="center">
    <kbd><img width="700" alt="Boxplot" src="https://github.com/fauzanheryka/Predict-Clicked-Ads-Customer-Classification-by-using-Machine-Learning/assets/141212116/f361601a-b060-4c57-a60e-56d558cb8a90"></kbd><br>
    Figure 15 - Confusion Matrix Logreg
</p>

### Insight : 
- It can be seen that the regression model can differentiate predicted cases of clicking on an ad even though they did not click on the ad as many as 143 in the test data (TN), this looks good for saving advertising costs and also does not disturb the experience of users who are not interested.
- There were 8 who were predicted not to click on ads and in fact they actually clicked on ads (FN), this looks good because it can prevent loss of revenue opportunities
- There are 3 users who are predicted to click on ads even though they don't actually want to. This looks good because it can prevent cost loss. (FP)
- There are 146 users who are predicted to click on ads and in fact actually want to click on ads, this looks good because they can get revenue and can provide effective marketing (TP).

<br>

<p align="center">
    <kbd><img width="700" alt="Boxplot" src="https://github.com/fauzanheryka/Predict-Clicked-Ads-Customer-Classification-by-using-Machine-Learning/assets/141212116/563e4025-ffe5-41ad-9183-ac9424fcb9af"></kbd><br>
    Figure 16 - Confusion Matrix KNN
</p>

We can see that both models are pretty good at predicting a case, and that the confusion matrix produced by KNN is not significantly different from that produced by logistic regression. As a result, because knn has large computing costs and is not robust against outliers, we will utilize **`logistic regression`** for this dataset, which has the advantage of being robust against outliers and easy to interpretation.

### Learning Curve 

<p align="center">
    <kbd><img width="700" alt="Boxplot" src="https://github.com/fauzanheryka/Predict-Clicked-Ads-Customer-Classification-by-using-Machine-Learning/assets/141212116/d5155268-6999-488d-888a-3b469f7f661f"></kbd><br>
    Figure 17 - Learning Curve
</p>

### Insight : 
The training and cross-validation scores grow as the number of training examples increases in this graph. However, after about 400 training examples, the cross-validation scores began to flatten. This signifies that the model has reached its limits and will not increase accuracy further with further training instances.
Some insights that can be gained:
- This model is great. The training and cross-validation scores are both high, and both improve as the number of training examples increases.
- This model has reached the limits of its capabilities. Its cross-validation scores flat after about 400 training examples.

<p align="center">
    <kbd><img width="900" alt="Boxplot" src="https://github.com/fauzanheryka/Predict-Clicked-Ads-Customer-Classification-by-using-Machine-Learning/assets/141212116/e0ca8ac6-909c-484f-a9b7-c7654f65fd28"></kbd><br>
    Figure 18 - Feature Importance
</p>

<p align="center">
    <kbd><img width="700" alt="Boxplot" src="https://github.com/fauzanheryka/Predict-Clicked-Ads-Customer-Classification-by-using-Machine-Learning/assets/141212116/a853731a-cb80-4360-91b3-563af3d4926c"></kbd><br>
    Figure 19 - Shap
</p>

## Business insight 
- On the `daily internet usage` feature,It can be seen that this feature has a strong correlation with the target however, based on eda, feature importance, and shape value, the feature has a negative influence, because the more frequently the customer uses the internet, the less likely he is to suppress advertising. Customers, on the other hand, tend to click adverts when they access the internet for a brief period of time. This could be because clients who use the internet heavily tend to be more productive or focused on specific tasks. Because they are focused on their activities, they may be less attentive to commercials, and as a result, there is less potential to click ads.
- In the `daily time spent on site` feature,As can be observed, this feature is similarly substantially connected with the target nevertheless, it has a noticeable detrimental influence on the resulting shap and coef. This feature, if connected to the eda, has something in common with the daily internet usage function, in that if users spend longer time on the website, they are less likely to click on adverts.
- The `area income` feature also appears to have a strong correlation but also has a negative influence, Although a high income frequently comes with greater purchasing capacity, spending habits and priorities also have a role. Some people with high salaries may be inclined to conserve rather than instantly spend their money on promoted goods or services.
- For `age` it also has an indication that it has a strong influence and has a positive influence on modeling, if you look at the eda and shape values, it shows that if the user's age is older, they are more likely to click on the ad. <br>

These four features greatly affect the model that the customer will click on the ad or not. This important feature will be used as a benchmark for business recommendations.

# STAGE 5. Business Recomendations and Simulation
## Business Recomendation 
After doing the modeling, we will then make a business recommendation based on eda, feature importance, shap :

- `Ad placements should be optimized` for active internet users. Ads are less effective with heavy internet users. Consider optimizing ad placement for users with limited internet usage or finding ways to make ads stand out for this demographic, such as through captivating images or unique offers that can raise user curiosity and eventual ad clicks.
- `Area income` Customers with high incomes do not tend to advertise, hence specific segmentation ads are required for all client categories in order to optimize the earnings acquired subsequently.
- `age` It is more dispersed among older individuals who click on ads, so it may be further optimized for material that can cover everything from young to senior, increasing the odds of customers being interested and clicking on the ad.

<br>

## Business Simulation 
When compared to the actual data, around 3.7% of the predictions were inaccurate. In other words, around 11 user of the 300 test data were incorrectly predicted by the model. As a result, the model can be considered to be **`highly good at classification`**.

### Without model
In this scheme, simulation is applied in data test with 300 user.<br>
For this simulation, I will use a source from [Signalmagz.com](https://www.sinyalmagz.com/tips-simulasi-menghitung-biaya-digital-marketing/)<br>

The data used in this business simulation is the data from the (data test). There are two schemes in this business simulation, namely `without machine learning` and `with machine learning`. With the assumptions used are as follows:

Advertising costs per customer = Rp. 1,000
Profit earned when a customer clicks on an ad = Rp. 5,000

`Without machine learning` user who receive adverts without model 300 user
, it was found that the CTR was 51,3% with a total cost of Rp. 300,000, revenue of Rp. 770,000 and the profit earned is Rp. 470,000.

`With machine learning` user who receive adverts is 154 it was found that the CTR was 95% with a total cost of Rp. 154,000, revenue of Rp. 770,000 and the profit earned is Rp. 616,000, So the machine learning model may be stated to be better because it is considerably `more efficient in terms of CTR and overall cost` than if you weren't using the model. **For the detail simulation, please check notebook**

# Conclusion 
- Without the CTR model, only 51% can be achieved, but with the CTR model it can reach 95%. This clearly shows a significant increase of around 44%.
- We can only do advertisements to all clients without exception without a model, which may result in potential cost loss, but if we use a model, we can save 49% on costs, which of course looks excellent so that we can maximize the profits we obtain later.
- It can be observed that the revenue obtained with or without the model is the same, but if the expenditures incurred are fewer, the profit will be maximized.
- The model is capable of increasing profit by 31% by making accurate model predictions with an error rate of 3.7%.