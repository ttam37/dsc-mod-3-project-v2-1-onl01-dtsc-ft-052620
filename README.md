# Predicting NBA Wins Classification

![](https://github.com/ttam37/dsc-mod-3-project-v2-1-onl01-dtsc-ft-052620/blob/master/images/nba_wallpaper.jpg)

Basketball has been around for longer than a century, while NBA has been around since 1946. There is a vast amount of NBA data and we are fortunate to have all this data easily accessible to everyone. There are a lot of resources online such as Basketball-Reference and the NBA official website. The goal of this classification model is predict NBA wins based on NBA statistics. You may be wondering what is the purpose of this model? Eventually I want my model to predict NBA wins based on spread, which is a term for betting purposes. Though this is only a preliminary model, I will need to interpret unsupervised learning into my model. In the meantime, classification is a good start and I will continue to implement this model in the future as I continue my journey to becoming a Data Scientist.


# Process Overview

![](https://github.com/ttam37/dsc-mod-3-project-v2-1-onl01-dtsc-ft-052620/blob/master/images/osemn_model.png)

# Technologies Used

## Python
* Pandas for Data Cleaning & Data Manipulation
* Matplotlib, Seaborn for Data Visualization
* Numpy for Data Calculations
* Scikit Learn for Model Algorithms

# Description of Dataset

The dataset is scraped by Nathan Lauga and is uploaded on Kaggle. He scraped the data from the NBA stats website using the NBA API. The datasets are from 2004 to 2019 season (first 66 games of 2019 due to date scraped). 

Kaggle website can be found here: https://www.kaggle.com/nathanlauga/nba-games

Description of each dataset:
* **games.csv** : all games from 2004 season to last update with the date, teams and some details like number of points, etc.
* **games_details.csv** : details of games dataset, all statistics of players for a given game
* **players.csv** : players details (name)
* **ranking.csv** : ranking of NBA given a day (split into west and east on CONFERENCE column
* **teams.csv** : all teams of NBA


# Exploratory Data Analysis

## 2004-2018 vs 2019 (First 66 Games) FG, 3FG, FT Statistics

![](https://github.com/ttam37/dsc-mod-3-project-v2-1-onl01-dtsc-ft-052620/blob/master/images/percentage_statistics.png)

## 2004-2018 vs 2019 (First 66 Games) PTS, REB, AST Statistics

![](https://github.com/ttam37/dsc-mod-3-project-v2-1-onl01-dtsc-ft-052620/blob/master/images/general_statistics.png)

## 2004 - 2019 (FIrst 66 Games) Statistics over the Seasons

![](https://github.com/ttam37/dsc-mod-3-project-v2-1-onl01-dtsc-ft-052620/blob/master/images/statistics_2004_2019_lineplot.png)

# Modeling

The algorithms that were used:
* Logistic Regression
* Decision Trees
* Random Forest
* K-Nearest Neighbors
* XGBoost
* Gradient Boosting
* LinearSVC

![](https://github.com/ttam37/dsc-mod-3-project-v2-1-onl01-dtsc-ft-052620/blob/master/images/ROC_curve.png)


# Final Model Results

**Best Model**: Linear SVC
**Final Accuracy Score**: 89.74%

![](https://github.com/ttam37/dsc-mod-3-project-v2-1-onl01-dtsc-ft-052620/blob/master/images/LinearSVC_metrics.png)
![](https://github.com/ttam37/dsc-mod-3-project-v2-1-onl01-dtsc-ft-052620/blob/master/images/LinearSVC_confusion_matrix_result.png)
![](https://github.com/ttam37/dsc-mod-3-project-v2-1-onl01-dtsc-ft-052620/blob/master/images/LinearSVC_confusion_matrix_result1.png)




