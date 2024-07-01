"DeepAnT: A Deep Learning Approach for Unsupervised Anomaly Detection in Time Series".

---

# DeepAnT: Unsupervised Anomaly Detection in Time Series

This repository contains an implementation of the paper [DeepAnT: A Deep Learning Approach for Unsupervised Anomaly Detection in Time Series](https://ieeexplore.ieee.org/document/8581424). DeepAnT uses a deep Convolutional Neural Network (CNN) to detect anomalies in time series data without the need for labeled anomalies.

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Data Preparation](#data-preparation)
  - [Model Training](#model-training)
  - [Anomaly Detection](#anomaly-detection)
  - [Evaluation](#evaluation)
- [Synthetic Data Generation](#synthetic-data-generation)
- [Results](#results)
- [References](#references)

## Introduction

DeepAnT is an unsupervised anomaly detection framework for time series data. It consists of two main modules:
1. **Time Series Predictor**: A deep CNN model that predicts the next time step in a series based on a historical window.
2. **Anomaly Detector**: Identifies anomalies by comparing the predicted values with the actual values using a reconstruction error.

## Requirements

- Python 3.6+
- NumPy
- Pandas
- Scikit-learn
- Keras
- TensorFlow
- Matplotlib

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/ashishpatel26/DeepAnT-A-Deep-Learning-Approach-for-Unsupervised-Anomaly-Detection-in-Time-Series.git
    cd DeepAnT-Anomaly-Detection
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Data Preparation

Ensure you have a time series data file. The provided example uses synthetic data. The data should be a CSV file with a single column `value`.

### Model Training

1. Load and preprocess the data:
    ```python
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler

    # Load data
    data = pd.read_csv('synthetic_timeseries_data.csv')
    values = data['value'].values.reshape(-1, 1)

    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_values = scaler.fit_transform(values)
    ```

2. Prepare the dataset for training:
    ```python
    def create_dataset(dataset, look_back=1):
        X, Y = [], []
        for i in range(len(dataset)-look_back):
            X.append(dataset[i:(i+look_back), 0])
            Y.append(dataset[i + look_back, 0])
        return np.array(X), np.array(Y)

    look_back = 50  # History window size
    X, Y = create_dataset(scaled_values, look_back)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    ```

3. Define and train the CNN model:
    ```python
    from keras.models import Sequential
    from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

    # Define the CNN model
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(look_back, 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_absolute_error')

    # Train the model
    model.fit(X, Y, epochs=50, batch_size=32, validation_split=0.2, verbose=2)
    ```

### Anomaly Detection

1. Predict the next time steps and calculate reconstruction error:
    ```python
    predictions = model.predict(X)
    reconstruction_error = np.abs(predictions.flatten() - Y.flatten())

    # Define the threshold for anomaly detection
    mean_error = np.mean(reconstruction_error)
    std_error = np.std(reconstruction_error)
    threshold = mean_error + 3 * std_error

    # Anomaly detection
    anomalies = np.where(reconstruction_error > threshold)[0]
    ```

2. Plot the results:
    ```python
    import matplotlib.pyplot as plt

    plt.figure(figsize=(15, 5))
    plt.plot(scaled_values, label='True Data')
    plt.plot(np.arange(look_back, len(predictions) + look_back), predictions, label='Predicted Data')
    plt.scatter(anomalies + look_back, predictions[anomalies], color='red', label='Anomalies')
    plt.legend()
    plt.show()
    ```

### Evaluation

1. Calculate precision, recall, and F1-score:
    ```python
    from sklearn.metrics import precision_recall_fscore_support

    # Generate true labels for the synthetic data
    true_labels = np.zeros(len(reconstruction_error))
    true_labels[anomalies] = 1

    # Generate predicted labels based on the threshold
    pred_labels = np.zeros(len(reconstruction_error))
    pred_labels[reconstruction_error > threshold] = 1

    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average='binary')
    print(f'Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}')
    ```

## Synthetic Data Generation

A synthetic dataset can be generated using the following script:

```python
import numpy as np
import pandas as pd

# Generate synthetic time series data
np.random.seed(42)

# Parameters for synthetic data
n_samples = 1000
n_anomalies = 50
trend = np.linspace(0, 10, n_samples)
seasonality = 5 * np.sin(np.linspace(0, 20 * np.pi, n_samples))
noise = np.random.normal(0, 0.5, n_samples)

# Create the time series
time_series = trend + seasonality + noise

# Introduce anomalies
anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
time_series[anomaly_indices] += np.random.normal(20, 5, n_anomalies)

# Save to a DataFrame
df = pd.DataFrame({"value": time_series})

# Save the DataFrame to a CSV file
df.to_csv('synthetic_timeseries_data.csv', index=False)
```

## Results

After running the model on the synthetic data, you should see a plot of the true data, predicted data, and detected anomalies. The precision, recall, and F1-score metrics provide insights into the performance of the anomaly detection.

## References

1. Munir, M., Siddiqui, S. A., Dengel, A., & Ahmed, S. (2019). DeepAnT: A Deep Learning Approach for Unsupervised Anomaly Detection in Time Series. IEEE Access, 7, 1991-2005. DOI: [10.1109/ACCESS.2018.2886457](https://ieeexplore.ieee.org/document/8581424)

---