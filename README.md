# Multimodal House Price Prediction App

This project is a **Streamlit web application** for predicting house prices using **both tabular data and images**. The app integrates a **preprocessed tabular pipeline** (with numeric scaling and categorical encoding) and **image embeddings from a pre-trained ResNet50**. Users can input tabular features and upload house images to get a predicted price.

---

## Features

- **Tabular Feature Pipeline**  
  - Numeric features: scaled with `MinMaxScaler`  
  - Categorical features: encoded using `OrdinalEncoder`  
  - Supports dynamic dropdowns for categorical columns based on the fitted pipeline  

- **Image Feature Pipeline**  
  - Extracts embeddings from house images using a frozen `ResNet50` model  
  - Combines image embeddings with preprocessed tabular features  

- **Prediction**  
  - Combines tabular and image features to predict house price using an `XGBoost` regressor  

- **Streamlit Frontend**  
  - User-friendly form for numeric, categorical, and image inputs  
  - Displays preprocessed tabular features and predicted price  

---

## Installation

1. Clone this repository:

```bash
git clone https://github.com/yourusername/multimodal-house-price.git
cd multimodal-house-price
````

2. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

> Requirements should include: `streamlit`, `tensorflow`, `keras`, `numpy`, `pandas`, `xgboost`, `scikit-learn`, `joblib`

---

## Project Structure

```
.
├── app.py                   # Streamlit app
├── models/
│   └── model.joblib          # Trained XGBoost model
├── pipelines/
│   └── preprocessor.joblib   # Tabular preprocessing pipeline
├── Data/
│   └── socal2/               # Image folder
│       └── <image_id>.jpg
├── requirements.txt
└── README.md
```

---

## Usage

1. Run the Streamlit app:

```bash
streamlit run app.py
```

2. In the app:

   * Enter numeric features (bedrooms, bathrooms, square footage, garage, etc.)
   * Select categorical features dynamically (e.g., city)
   * Upload a house image (`.jpg`, `.jpeg`, `.png`)
   * Click **Predict Price**

3. The app will display:

   * **Preprocessed tabular features** (including encoded categorical features)
   * **Predicted house price**

---

## How It Works

1. **Tabular Preprocessing**

   * Numeric columns are imputed and scaled.
   * Categorical columns are imputed and encoded with `OrdinalEncoder`.
   * Feature names are extracted from the fitted pipeline for dynamic display in Streamlit.

2. **Image Embeddings**

   * Uploaded images are temporarily stored using `tempfile` (Windows-safe).
   * Images are resized and normalized, then passed through a frozen `ResNet50` model.
   * Embeddings are extracted using global average pooling.

3. **Prediction**

   * Tabular and image features are concatenated.
   * The trained `XGBoost` regressor predicts the house price.

---

## Notes

* Ensure all **images** are named according to the dataset's `image_id` column if batch predictions are required.
* The **city selectbox** is automatically populated based on the fitted `OrdinalEncoder`, so only valid categories can be selected.
* Temporary files for uploaded images are safely deleted after processing to avoid permission issues on Windows.


```
