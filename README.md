***

# Time-Series Forecasting of PM2.5 and PM10 Using LSTM Networks

## 📄 Abstract
Air pollution is a critical public health concern, particularly in urban environments like Delhi. This project implements a Deep Learning approach using Long Short-Term Memory (LSTM) networks to forecast the concentration of hazardous particulate matter (PM2.5 and PM10). By analyzing historical Air Quality Index (AQI) data, the model captures temporal dependencies to provide accurate short-term predictions, assisting in early warning systems and decision-making,.

## 📋 Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Methodology](#methodology)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Limitations & Future Scope](#limitations--future-scope)
- [References](#references)

## 📖 Introduction
Traditional statistical models (like ARIMA) often struggle with the non-linear complexities of environmental data. This project leverages Recurrent Neural Networks (RNNs), specifically LSTMs, which are superior in handling time-series forecasting tasks. The objective is to predict future PM2.5 and PM10 levels based on multivariate historical data,.

## 📊 Dataset
The model is trained on historical air quality measurements from **Delhi**. The dataset is a multivariate time-series containing the following features,:

*   **Timestamp:** `date` (hourly observations)
*   **Target Variables:** `pm2_5`, `pm10`
*   **Input Features:**
    *   `co` (Carbon Monoxide)
    *   `no`, `no2` (Nitric Oxide, Nitrogen Dioxide)
    *   `o3` (Ozone)
    *   `so2` (Sulfur Dioxide)
    *   `nh3` (Ammonia)

**Data Processing:**
*   Missing values are handled using the forward fill method (`ffill`),.
*   Data is normalized using **Min-Max Scaling** to map values between 0 and 1,.

## 🛠 Tech Stack
The project is implemented in **Python** using the following libraries,:
*   **Data Handling:** `Pandas`, `NumPy`
*   **Deep Learning:** `TensorFlow`, `Keras`
*   **Preprocessing:** `Scikit-learn` (MinMaxScaler)
*   **Visualization:** `Matplotlib`

## ⚙️ Methodology

### 1. Feature Engineering
The model utilizes a **sliding window technique** (Lookback mechanism). The inputs are sequences of the previous **24 time steps** (hours) used to predict the pollution levels of the subsequent step,.

### 2. Model Architecture
The LSTM model is built using the Keras Sequential API with the following structure,:
*   **Input Layer:** LSTM with **64 units** (returns sequences).
*   **Regularization:** Dropout layer (**0.2**) to prevent overfitting.
*   **Hidden Layer:** LSTM with **32 units**.
*   **Output Layer:** Dense layer with **2 neurons** (predicting PM2.5 and PM10).
*   **Optimizer:** Adam.
*   **Loss Function:** Mean Squared Error (MSE).

### 3. Training
*   **Split:** 80% Training, 20% Testing (preserving temporal order to prevent data leakage),.
*   **Epochs:** 40
*   **Batch Size:** 32.

## 📈 Results

The model performance is evaluated using **Root Mean Squared Error (RMSE)** and **Mean Absolute Error (MAE)**.

*   **Performance:** The LSTM effectively learns temporal patterns, with predicted values closely tracking actual observations during stable periods.
*   **Visualization:** The repository includes line plots comparing `Actual` vs. `Predicted` values for both PM2.5 and PM10, illustrating the model's ability to capture diurnal cycles (e.g., morning rise, evening peak),.

*Note: While the LSTM model performs well, alternative experiments using XGBoost were also conducted, highlighting strong diurnal patterns in the data such as morning traffic spikes and midday dips,.*

## ⚠️ Limitations & Future Scope
### Limitations
*   **External Factors:** Meteorological data (wind speed, temperature) were not included in this iteration, limiting the model's ability to predict pollution dispersion.
*   **Spikes:** The model occasionally deviates during sudden, extreme pollution events.

### Future Scope
*   **Data Integration:** Incorporate weather and traffic volume data to improve robustness.
*   **Advanced Architectures:** Experiment with **ConvLSTM** or Attention-based mechanisms to better capture spatiotemporal dependencies,.
*   **Deployment:** Develop a real-time web dashboard for public monitoring.

## 📚 References
This project draws inspiration from:
1.  **Hochreiter, S., & Schmidhuber, J. (1997).** Long Short-Term Memory. *Neural Computation*.
2.  **Dai, Y., et al. (2025).** High-resolution climate prediction... using ConvLSTM-XGBoost.
3.  **Li, J., et al. (2022).** Air quality indicators and AQI prediction coupling LSTM and SSA.

---
*Created for the Air Quality Prediction Project Report.*
