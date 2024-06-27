# CDC Diabetes Indicators Analysis

## Project Overview

This project analyzes the CDC Diabetes Indicators dataset from the UCI Machine Learning Repository. The goal is to explore the data, build predictive models, and gain insights into diabetes indicators.

## Dataset

The dataset can be found [here](https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008).

## Repository Structure
```bash
├── data
│   ├── diabetes.csv
│   ├── X_train.csv
│   ├── X_test.csv
│   ├── y_train.csv
│   └── y_test.csv
├── models
│   └── logistic_regression_model.pkl
├── notebooks
│   └── EDA.py
├── scripts
│   ├── data_preprocessing.py
│   ├── train_model.py
│   └── evaluate_model.py
├── README.md
├── requirements.txt
└── .gitignore
```



## Getting Started

### Prerequisites

- Python 3.x
- VSCode

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/CDC-Diabetes-Indicators-Analysis.git
   cd CDC-Diabetes-Indicators-Analysis

python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
    Add the following libraries to requirements.txt to install
    pandas
    numpy
    matplotlib
    seaborn
    scikit-learn
    jupyter

### Running the Project

Open the 'EDA.py' script in VSCode.
Run the script to perform exploratory data analysis.

### Contributing

Feel free to submit issues or pull requests if you find any bugs or have suggestions for improvements.

License

This project is licensed under the MIT License.

Let me know if you need help with any specific part of the process!
