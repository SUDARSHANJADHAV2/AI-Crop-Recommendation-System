# KrushiAI - Smart Crop Recommendation System

KrushiAI is an intelligent crop recommendation system that leverages machine learning to help farmers make informed decisions about crop selection. By analyzing soil composition and environmental factors, this application suggests the most suitable crops to plant, leading to better yields and more sustainable farming practices.

## Live Demo

You can access the live application here:
[https://krushiai-crop-recommendation-system.streamlit.app/](https://krushiai-crop-recommendation-system.streamlit.app/)

## Key Features

-   **Crop Recommendation:** Get personalized crop recommendations based on soil and climate data.
-   **User-Friendly Interface:** A simple and intuitive web interface built with Streamlit.
-   **Data-Driven Insights:** The recommendations are powered by a robust Random Forest model.
-   **Detailed Information:** Provides details about the recommended crop and its ideal growing conditions.

## Tech Stack

-   **Python:** The core programming language.
-   **Pandas & NumPy:** For data manipulation and numerical operations.
-   **Scikit-learn:** For building and evaluating machine learning models.
-   **Streamlit:** For creating and deploying the web application.
-   **Jupyter Notebook:** For model development and experimentation.

## Project Structure

```
├── assets/
│   ├── crop.png
├── data/
│   ├── Crop_recommendation.csv
├── models/
│   ├── DecisionTree.pkl
│   ├── KNeighborsClassifier.pkl
│   ├── NBClassifier.pkl
│   ├── RF.pkl
│   ├── RandomForest.pkl
│   └── XGBoost.pkl
├── KrushiAI_Crop_Recommendation.ipynb
├── README.md
├── requirements.txt
└── webapp.py
```

## Dataset

The model was trained on the `Crop_recommendation.csv` dataset, which contains 2200 data points. The dataset has the following 8 columns:
-   `N`: Nitrogen content in soil
-   `P`: Phosphorus content in soil
-   `K`: Potassium content in soil
-   `temperature`: Temperature in Celsius
-   `humidity`: Relative humidity in %
-   `ph`: pH value of the soil
-   `rainfall`: Rainfall in mm
-   `label`: The recommended crop (22 unique crop types)

## Model Training and Evaluation

Several machine learning models were trained and evaluated to find the best one for this task. The Random Forest model was chosen for the final application due to its high accuracy of **99.5%** on the test set.

**Classification Report:**

| Crop          | Precision | Recall | F1-Score | Support |
|---------------|-----------|--------|----------|---------|
| apple         | 1.00      | 1.00   | 1.00     | 13      |
| banana        | 1.00      | 1.00   | 1.00     | 17      |
| blackgram     | 1.00      | 1.00   | 1.00     | 16      |
| chickpea      | 1.00      | 1.00   | 1.00     | 21      |
| coconut       | 1.00      | 1.00   | 1.00     | 21      |
| coffee        | 1.00      | 1.00   | 1.00     | 22      |
| cotton        | 1.00      | 1.00   | 1.00     | 20      |
| grapes        | 1.00      | 1.00   | 1.00     | 18      |
| jute          | 0.93      | 1.00   | 0.97     | 28      |
| kidneybeans   | 1.00      | 1.00   | 1.00     | 14      |
| lentil        | 1.00      | 1.00   | 1.00     | 23      |
| maize         | 1.00      | 1.00   | 1.00     | 21      |
| mango         | 1.00      | 1.00   | 1.00     | 26      |
| mothbeans     | 1.00      | 1.00   | 1.00     | 19      |
| mungbean      | 1.00      | 1.00   | 1.00     | 24      |
| muskmelon     | 1.00      | 1.00   | 1.00     | 23      |
| orange        | 1.00      | 1.00   | 1.00     | 29      |
| papaya        | 1.00      | 1.00   | 1.00     | 19      |
| pigeonpeas    | 1.00      | 1.00   | 1.00     | 18      |
| pomegranate   | 1.00      | 1.00   | 1.00     | 17      |
| rice          | 1.00      | 0.88   | 0.93     | 16      |
| watermelon    | 1.00      | 1.00   | 1.00     | 15      |
| **Accuracy**  |           |        | **1.00** | **440** |
| **Macro Avg** | 1.00      | 0.99   | 1.00     | 440     |
| **Weighted Avg**| 1.00    | 1.00   | 1.00     | 440     |

**Confusion Matrix:**

![Confusion Matrix](confusion_matrix.png)

## How to Run Locally

To run this project on your local machine, follow these steps:

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/your-username/KrushiAI-Crop-Recommendation-System.git
    cd KrushiAI-Crop-Recommendation-System
    ```

2.  **Create a virtual environment:**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit application:**
    ```sh
    streamlit run webapp.py
    ```

## How to Contribute

Contributions are welcome! If you have any ideas, suggestions, or bug reports, please open an issue. If you want to contribute code, please fork the repository and create a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
