# Machine Learning for Predictive Maintenance

I built this project to predict machine failures from operating data such as temperature, rotational speed, torque and tool wear.

The project uses the [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset), which contains 10,000 synthetic machine records.

## What I did

* Explored and cleaned the data using pandas
* Compared normal and failed machines through visualisations
* Prepared the data for machine learning
* Trained Logistic Regression and Random Forest models
* Compared the models using precision, recall and F1-score
* Checked which features were most important
* Saved the trained Random Forest model with joblib

## Results

Only 3.39% of the records are failures, so accuracy alone can be misleading. I focused mainly on the results for the failure class.

| Model               | Precision | Recall | F1-score |
| ------------------- | --------: | -----: | -------: |
| Logistic Regression |      0.14 |   0.82 |     0.24 |
| Random Forest       |      0.70 |   0.65 |     0.67 |

Logistic Regression detected more failures, but it also produced many false alarms. Random Forest gave a much better overall balance, increasing the failure F1-score from 0.24 to 0.67.

## What the model learned

The three most important features were:

1. Rotational speed: 30.8%
2. Torque: 29.7%
3. Tool wear: 20.7%

Together, they accounted for around 81.1% of the model’s total feature importance. This fits the failure patterns represented in the dataset, where machine failures are closely related to speed, torque and tool wear.

<img width="650" alt="FeatureImportance" src="https://github.com/user-attachments/assets/afd7aafe-b777-498c-b61f-71ea2b4901c4" />

Feature importance shows what the model relied on for prediction, but it does not prove that a feature directly caused a failure.

## Project Files

* `analysis.ipynb` — data analysis, model training and evaluation
* `data/ai4i2020.csv` — dataset
* `model.joblib` — saved Random Forest model

## Run the Project

Install the required packages:

```bash
pip install pandas matplotlib seaborn scikit-learn joblib jupyter
```

Then open the notebook:

```bash
jupyter notebook analysis.ipynb
```

Run the cells from top to bottom to reproduce the results.

## Next Steps

I plan to add cross-validation, improve the model evaluation and build a simple API for making predictions.
