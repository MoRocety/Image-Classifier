# Image Classifier Project

This repository contains various Python scripts for image classification and feature extraction. The project includes multiple tasks and models for different purposes, such as age prediction, expression recognition, and face classification.

## Table of Contents

- [Setup Instructions](#setup-instructions)
- [Tasks and Models](#tasks-and-models)
  - [Assignment 2](#assignment-2)
    - [Task 1](#task-1)
    - [Task 2](#task-2)
    - [Task 3](#task-3)
  - [Assignment 3](#assignment-3)
    - [Task 1](#task-1-1)
    - [Task 2](#task-2)
- [Models](#models)

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/MoRocety/Image-Classifier.git
   cd Image-Classifier
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Tasks and Models

### Assignment 2

#### Task 1

- **Description**: This task involves training a linear regression model for age prediction using image features.
- **Script**: `Assignment2/task1.py`
- **Model**: `Assignment2/linear_regression_model.pkl`

#### Task 2

- **Description**: This task involves training a SGD regression model and comparing its performance with the linear regression model from Task 1.
- **Script**: `Assignment2/task2.py`
- **Model**: `Assignment2/sgd_regression_model.pkl`

#### Task 3

- **Description**: This task involves using the trained models for real-time age prediction using a webcam.
- **Script**: `Assignment2/task3.py`

### Assignment 3

#### Task 1

- **Description**: This task involves training and evaluating various regression models, including Ridge, ElasticNet, Lasso, and SGD, for age prediction.
- **Script**: `Assignment3/task1/task1.py`
- **Models**: 
  - `Assignment3/task1/ridge_model.pkl`
  - `Assignment3/task1/elastic_net_model.pkl`
  - `Assignment3/task1/lasso_model.pkl`
  - `Assignment3/task1/sgd_regression_model.pkl`
  - `Assignment3/task1/sgd_regression_optimal_model.pkl`
  - `Assignment3/task1/best_model.pkl`

#### Task 2

- **Description**: This task involves training and evaluating logistic regression models for expression recognition and name classification.
- **Script**: `Assignment3/task2/task2.py`
- **Models**: 
  - `Assignment3/task2/logistic_reg_expression_model.pkl`
  - `Assignment3/task2/logistic_reg_name_model.pkl`

## Models

The repository includes various models used in the project:

- **Models**:
  - `Assignment2/linear_regression_model.pkl`
  - `Assignment2/sgd_regression_model.pkl`
  - `Assignment3/task1/ridge_model.pkl`
  - `Assignment3/task1/elastic_net_model.pkl`
  - `Assignment3/task1/lasso_model.pkl`
  - `Assignment3/task1/sgd_regression_model.pkl`
  - `Assignment3/task1/sgd_regression_optimal_model.pkl`
  - `Assignment3/task1/best_model.pkl`
  - `Assignment3/task2/logistic_reg_expression_model.pkl`
  - `Assignment3/task2/logistic_reg_name_model.pkl`
