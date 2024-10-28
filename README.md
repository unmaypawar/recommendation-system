# recommendation-system

## Description

<p>This project aims to develop a weighted hybrid recommendation system utilizing the Yelp dataset, focusing on improving the Root Mean Square Error (RMSE) of ratings predictions. The primary data source includes user IDs, business IDs, and ratings, while additional features are engineered from various related datasets. Effective feature engineering is crucial, as the selection of relevant attributes significantly impacts model performance.</p>

<p>Key features were extracted and transformed to enhance the recommendation model. Attributes related to user behavior were processed using binary encoding or retained as numerical values. Various engagement metrics, aggregates of check-ins, and counts of associated photos were incorporated into the feature set. Additionally, likes on user tips were included, and a heatmap of feature correlations with ratings was generated to guide the attribute selection process.</p>

<p>The final model employs XGBoost and CatBoost as the chosen regressor models. Each feature was added iteratively to assess its effect on RMSE, ensuring that only the most impactful attributes were retained. This study provides insights into the importance of feature selection and model integration, ultimately aiming to enhance the efficacy of recommendation systems in user-driven platforms.</p>

## Installation Instructions

```
docker build -t data_mining_image .
```

```
docker run -it --name data_mining_container -p 8888:8888 -v absolute/path/to/project/folder:/workspace data_mining_image /bin/bash 
```

## Usage Instructions

### Dataset

Download yelp dataset from [here](https://www.yelp.com/dataset)

### Execution

```
spark-submit main.py data path/to/test/file path/to/output/file
```

## Contribution Guidelines

Fork the repository to your GitHub account and please create a new branch for each feature or bug fix.

## Support Information

Please raise an issue or a pull request.