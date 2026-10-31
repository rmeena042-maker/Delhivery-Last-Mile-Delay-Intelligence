# Delhivery Last-Mile Delay Intelligence Pipeline

## Project Overview
This project builds an end-to-end AI/ML pipeline to analyze and predict last-mile delivery delays.

It uses shipment data, hub information, regression, classification and clustering techniques to support better delivery operations.

## Project Objectives
- Clean and prepare shipment data.
- Perform exploratory data analysis.
- Predict delivery time using regression models.
- Classify shipments as delayed or not delayed.
- Segment delivery hubs into Standard, Watch and Escalate categories.
- Generate delay predictions and recommended actions for new shipments.

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib
- Git and GitHub

## Project Structure

- data/ — Input datasets and modelling table
- models/ — Saved machine learning models
- outputs/ — Analysis results and prediction reports
- src/ — Data preparation and ML scripts
- generate_data.py — Dataset generation
- part_a_clean_eda.py — Data cleaning and EDA
- package_model.py — Model packaging
- predict_delay.py — New shipment predictions

## How to Run

Install dependencies:

pip install -r requirements.txt

Generate data:

python generate_data.py

Run data preparation:

python part_a_clean_eda.py

Run regression:

python src/part_b_regression.py

Run classification:

python src/part_c_classification.py

Run hub segmentation:

python src/part_d_segments.py

Package the model:

python package_model.py

Predict new shipments:

python predict_delay.py

## Prediction Output

The pipeline generates new_shipment_actions.csv containing:
- Shipment ID
- Hub ID
- Hub segment
- Delay probability
- Flagged status
- Recommended action

## Author
Dushyant Kumar Rahul
