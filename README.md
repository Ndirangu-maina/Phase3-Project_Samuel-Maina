# Phase3-Project_Samuel-Maina
## Business Understanding
The real estate market requires accurate pricing predictions to help sellers set competitive prices
and buyers make informed decisions. This dataset contains detailed features of residential homes
in Ames, Iowa, including physical characteristics, location details, and sale prices.
The goal of this project is to analyze various factors influencing house prices in these locations and develop predictive models to estimate house prices based on these factors.
## Problem Statement
To develop a predictive model to accurately estimate house sale prices based on various property
features, enabling stakeholders to make data-driven pricing decisions and identify key factors
influencing property values
## Objectives
1. Build an accurate predictive model for house prices.
2. Identify key features influencing house prices.
3. Provide actionable insights for real estate stakeholders.
4. Compare different machine learning models for optimal performance.
## Data Understanding
Dataset Shape: (1460, 81)

First few rows:
    Id  MSSubClass MSZoning  LotFrontage  LotArea Street Alley LotShape  \
0   1          60       RL         65.0     8450   Pave   NaN      Reg   
1   2          20       RL         80.0     9600   Pave   NaN      Reg   
2   3          60       RL         68.0    11250   Pave   NaN      IR1   
3   4          70       RL         60.0     9550   Pave   NaN      IR1   
4   5          60       RL         84.0    14260   Pave   NaN      IR1   

  LandContour Utilities  ... PoolArea PoolQC Fence MiscFeature MiscVal MoSold  \
0         Lvl    AllPub  ...        0    NaN   NaN         NaN       0      2   
1         Lvl    AllPub  ...        0    NaN   NaN         NaN       0      5   
2         Lvl    AllPub  ...        0    NaN   NaN         NaN       0      9   
3         Lvl    AllPub  ...        0    NaN   NaN         NaN       0      2   
4         Lvl    AllPub  ...        0    NaN   NaN         NaN       0     12   

  YrSold  SaleType  SaleCondition  SalePrice  
0   2008        WD         Normal     208500  
1   2007        WD         Normal     181500  
2   2008        WD         Normal     223500  
3   2006        WD        Abnorml     140000  
4   2008        WD         Normal     250000  

[5 rows x 81 columns]

Data Info:

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1460 entries, 0 to 1459
Data columns (total 81 columns):
 #   Column         Non-Null Count  Dtype  
---  ------         --------------  -----  
 0   Id             1460 non-null   int64  
 1   MSSubClass     1460 non-null   int64  
 2   MSZoning       1460 non-null   object 
 3   LotFrontage    1201 non-null   float64
 4   LotArea        1460 non-null   int64  
 5   Street         1460 non-null   object 
 6   Alley          91 non-null     object 
 7   LotShape       1460 non-null   object 
 8   LandContour    1460 non-null   object 
 9   Utilities      1460 non-null   object 
 10  LotConfig      1460 non-null   object 
 11  LandSlope      1460 non-null   object 
 12  Neighborhood   1460 non-null   object 
 13  Condition1     1460 non-null   object 
 14  Condition2     1460 non-null   object 
 15  BldgType       1460 non-null   object 
 16  HouseStyle     1460 non-null   object 
 17  OverallQual    1460 non-null   int64  
 18  OverallCond    1460 non-null   int64  
 19  YearBuilt      1460 non-null   int64  
 20  YearRemodAdd   1460 non-null   int64  
 21  RoofStyle      1460 non-null   object 
 22  RoofMatl       1460 non-null   object 
 23  Exterior1st    1460 non-null   object 
 24  Exterior2nd    1460 non-null   object 
 25  MasVnrType     588 non-null    object 
 26  MasVnrArea     1452 non-null   float64
 27  ExterQual      1460 non-null   object 
 28  ExterCond      1460 non-null   object 
 29  Foundation     1460 non-null   object 
 30  BsmtQual       1423 non-null   object 
 31  BsmtCond       1423 non-null   object 
 32  BsmtExposure   1422 non-null   object 
 33  BsmtFinType1   1423 non-null   object 
 34  BsmtFinSF1     1460 non-null   int64  
 35  BsmtFinType2   1422 non-null   object 
 36  BsmtFinSF2     1460 non-null   int64  
 37  BsmtUnfSF      1460 non-null   int64  
 38  TotalBsmtSF    1460 non-null   int64  
 39  Heating        1460 non-null   object 
 40  HeatingQC      1460 non-null   object 
 41  CentralAir     1460 non-null   object 
 42  Electrical     1459 non-null   object 
 43  1stFlrSF       1460 non-null   int64  
 44  2ndFlrSF       1460 non-null   int64  
 45  LowQualFinSF   1460 non-null   int64  
 46  GrLivArea      1460 non-null   int64  
 47  BsmtFullBath   1460 non-null   int64  
 48  BsmtHalfBath   1460 non-null   int64  
 49  FullBath       1460 non-null   int64  
 50  HalfBath       1460 non-null   int64  
 51  BedroomAbvGr   1460 non-null   int64  
 52  KitchenAbvGr   1460 non-null   int64  
 53  KitchenQual    1460 non-null   object 
 54  TotRmsAbvGrd   1460 non-null   int64  
 55  Functional     1460 non-null   object 
 56  Fireplaces     1460 non-null   int64  
 57  FireplaceQu    770 non-null    object 
 58  GarageType     1379 non-null   object 
 59  GarageYrBlt    1379 non-null   float64
 60  GarageFinish   1379 non-null   object 
 61  GarageCars     1460 non-null   int64  
 62  GarageArea     1460 non-null   int64  
 63  GarageQual     1379 non-null   object 
 64  GarageCond     1379 non-null   object 
 65  PavedDrive     1460 non-null   object 
 66  WoodDeckSF     1460 non-null   int64  
 67  OpenPorchSF    1460 non-null   int64  
 68  EnclosedPorch  1460 non-null   int64  
 69  3SsnPorch      1460 non-null   int64  
 70  ScreenPorch    1460 non-null   int64  
 71  PoolArea       1460 non-null   int64  
 72  PoolQC         7 non-null      object 
 73  Fence          281 non-null    object 
 74  MiscFeature    54 non-null     object 
 75  MiscVal        1460 non-null   int64  
 76  MoSold         1460 non-null   int64  
 77  YrSold         1460 non-null   int64  
 78  SaleType       1460 non-null   object 
 79  SaleCondition  1460 non-null   object 
 80  SalePrice      1460 non-null   int64  
dtypes: float64(3), int64(35), object(43)
memory usage: 924.0+ KB

Basic Statistics:
                 Id   MSSubClass  LotFrontage        LotArea  OverallQual  \
count  1460.000000  1460.000000  1201.000000    1460.000000  1460.000000   
mean    730.500000    56.897260    70.049958   10516.828082     6.099315   
std     421.610009    42.300571    24.284752    9981.264932     1.382997   
min       1.000000    20.000000    21.000000    1300.000000     1.000000   
25%     365.750000    20.000000    59.000000    7553.500000     5.000000   
50%     730.500000    50.000000    69.000000    9478.500000     6.000000   
75%    1095.250000    70.000000    80.000000   11601.500000     7.000000   
max    1460.000000   190.000000   313.000000  215245.000000    10.000000   

       OverallCond    YearBuilt  YearRemodAdd   MasVnrArea   BsmtFinSF1  ...  \
count  1460.000000  1460.000000   1460.000000  1452.000000  1460.000000  ...   
mean      5.575342  1971.267808   1984.865753   103.685262   443.639726  ...   
std       1.112799    30.202904     20.645407   181.066207   456.098091  ...   
min       1.000000  1872.000000   1950.000000     0.000000     0.000000  ...   
25%       5.000000  1954.000000   1967.000000     0.000000     0.000000  ...   
50%       5.000000  1973.000000   1994.000000     0.000000   383.500000  ...   
75%       6.000000  2000.000000   2004.000000   166.000000   712.250000  ...   
max       9.000000  2010.000000   2010.000000  1600.000000  5644.000000  ...   

        WoodDeckSF  OpenPorchSF  EnclosedPorch    3SsnPorch  ScreenPorch  \
count  1460.000000  1460.000000    1460.000000  1460.000000  1460.000000   
mean     94.244521    46.660274      21.954110     3.409589    15.060959   
std     125.338794    66.256028      61.119149    29.317331    55.757415   
min       0.000000     0.000000       0.000000     0.000000     0.000000   
25%       0.000000     0.000000       0.000000     0.000000     0.000000   
50%       0.000000    25.000000       0.000000     0.000000     0.000000   
75%     168.000000    68.000000       0.000000     0.000000     0.000000   
max     857.000000   547.000000     552.000000   508.000000   480.000000   

          PoolArea       MiscVal       MoSold       YrSold      SalePrice  
count  1460.000000   1460.000000  1460.000000  1460.000000    1460.000000  
mean      2.758904     43.489041     6.321918  2007.815753  180921.195890  
std      40.177307    496.123024     2.703626     1.328095   79442.502883  
min       0.000000      0.000000     1.000000  2006.000000   34900.000000  
25%       0.000000      0.000000     5.000000  2007.000000  129975.000000  
50%       0.000000      0.000000     6.000000  2008.000000  163000.000000  
75%       0.000000      0.000000     8.000000  2009.000000  214000.000000  
max     738.000000  15500.000000    12.000000  2010.000000  755000.000000  

[8 rows x 38 columns]
**Dataset Overview:**

**Size and Structure**:The dataset contains 1460 rows and 81 columns.
The dataset includes a mix of numerical (int64, float64) and categorical (object) data types.

**Target Variable**:The primary target variable is "SalePrice," which indicates the price of the house.

**Missing Values**:A significant number of columns have missing values. Notably:"Alley," "PoolQC," "Fence," and "MiscFeature" have a very high percentage.
This indicates that data cleaning and imputation will be crucial steps.

**Data Types**:The dataset has 35 integer columns, 3 float columns, and 43 object (categorical) columns. This highlights the need for appropriate handling of both numerical and categorical features.


**Key Statistical Insights:**

***SalePrice Distribution:***
- The average sale price is approximately $180,921.

- The minimum sale price is $34,900.

- The maximum is $755,000, indicating a wide range of property values.
- The standard deviation of sale price is quite high, meaning that the sale prices are spread out.

***Overall Quality and Condition:***

- The "OverallQual" (overall quality) and "OverallCond" (overall condition) columns have a range from 1 to 10 and 1 to 9, respectively, providing a numerical assessment of property quality.
- The mean of OverallQual is 6.09, and the mean of OverallCond is 5.57.

***Year Built and Remodeled:***
The "YearBuilt" and "YearRemodAdd" columns show a range of construction and remodeling years, providing a timeline for property development.

***Lot Area and Lot Frontage:***
"LotArea" and "LotFrontage" provide information about the size and dimensions of the property. The large standard deviation of lot area shows that there is a large variety of lot sizes.

***Basement and Garage Features:***
Columns related to basement and garage features (e.g., "BsmtFinSF1," "GarageCars," "GarageArea") provide insights into the property's infrastructure.

***Porch and Deck Areas:***
"WoodDeckSF," "OpenPorchSF," and other porch-related columns indicate the presence and size of outdoor living spaces.

***Pool and Miscellaneous Features:***
The "PoolArea" and "MiscVal" columns indicate that pools and other miscellaneous features are relatively rare.

## EDA
### Univariate Analysis
![image](https://github.com/user-attachments/assets/98f26176-e75e-4bcb-ad6d-cdd80c589742)
The histogram provides a visual representation of the distribution of sale prices, highlighting its right-skewed nature, the central tendency, spread, and potential outliers.
The distribution of sale prices is right-skewed. This means that the tail of the distribution extends further to the right (higher prices), indicating that there are more houses with lower sale prices and fewer houses with very high sale prices.

### Bivariate Analysis
![image](https://github.com/user-attachments/assets/3de21963-eda4-4b72-89be-fdfcaee2240a)
The scatter plot reveals a positive correlation between living area and sale price. This means that, in general, houses with larger living areas tend to have higher sale prices.

The correlation appears to be moderately strong, but not perfect. While there's a clear trend, there's also some scatter, indicating that other factors besides living area also influence sale price.

### Multivariate Analysis
![image](https://github.com/user-attachments/assets/6f4a179c-5176-4afa-98f3-2c2ec2a4874b)
The correlation matrix heatmap above provides a visual overview of the relationships between numerical features in the dataset.

Some of the Key Observations are as below;

**Strong Positive Correlations:** You can identify strong positive correlations (dark red areas)
- "OverallQual" and "SalePrice" (as expected, higher quality houses sell for more).
- "GarageCars" and "GarageArea" (more garage space correlates with more cars).
- "GrLivArea" and "SalePrice" (larger living area correlates with higher price).
- "TotalBsmtSF" and "1stFlrSF" (a larger total basement square footage correlates with a larger 1st floor square footage).

## Preprocessing
### Encoding 
### Scaling
20% of our data is in test and 80% trains the model
Scaling, standardizes feature values so that some values are not penalized unfairly

## Modeling
### Classification
Model Performance:
Linear Regression: RMSE = 36658.46, R2 = 0.81
Decision Tree: RMSE = 38703.79, R2 = 0.79
Random Forest: RMSE = 28328.87, R2 = 0.89
K-NN: RMSE = 40809.63, R2 = 0.77
SVM: RMSE = 86886.76, R2 = -0.04

## Model Evaluation
With an RMSE of 28165.18 and R-Squared of 0.89, Random Forest model is definitely.

## Cocusion and Recommendation

From my objectives stated above;

**Below is my Conclusion:**
The Random Forest model with tuned hyperparameters provided the best performance for predicting house prices.
Key features influencing prices include TotalSF, GrLivArea, and OverallQual.

**Below are my Recommendations:**
1. Focus on property size and quality for pricing decisions.
2. Use the model for initial price estimates.
3. Future improvements could include using advanced ensemble methods and feature selection techniques.







