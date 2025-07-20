# Laptop Price Prediction
This project builds a machine learning model to predict laptop prices based on hardware and configuration features. The model is trained using a Random Forest Regressor and evaluated with multiple regression metrics.

# Project Structure
```
├── /data
│   └── laptop_price.csv        # Images to classify (not organized by class)
├── /notebooks
│   ├── data-analysis.ipynb     # Exploratory data analysis and dataset overview
│   └── model-training.ipynb    # Model training experiments and results visualization
├── /results                    # Folder for traind model and saved plots
├── /src
│   ├── requirements.txt/       # Required Libraries
│   └── train_model.py          # Main training, evaluation, and prediction script
```

# Installation
1. Clone the repository:
```bash
git clone https://github.com/Culetter/laptop-price-prediction.git
cd laptop-price-prediction/src
```
2. Install the dependencies
```
pip install -r requirements.txt
```

# Usage
To train the model and run predictions:
```
python train_model.py
```

# Features Used for Training
* Company
* TypeName
* ScreenResolution
* Cpu
* Ram
* Memory
* Gpu
* OpSys
* Weight

# Technologies
* pandas
* scickit-learn
* matplotlib
* joblib

# Evaluation Metrics
* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R² Score

# Author
**Nazarii Lozynskyi**  
[@Culetter](https://github.com/Culetter)

# License
The dataset used in this project is the "Intel Image Classification" dataset, available on Kaggle:  
https://www.kaggle.com/datasets/muhammetvarl/laptop-price

The dataset does not have a specific license listed, so it is used here only for educational and non-commercial purposes.  
All rights to the original dataset remain with the original author.
