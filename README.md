# recommendation-system

## Description



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