# 📉 Customer Churn Risk Predictor

## The Problem

A telecom company loses customers every month. Winning a new customer costs far more than keeping an existing one, so the company needs to **spot customers who are likely to leave (churn) early enough to give them a reason to stay**.

This app turns customer account data into a clear, actionable churn risk score that a retention team can use to decide who to contact first.

## Features

### Churn probability
Enter a customer's details and instantly get the probability (e.g. 65.1%) that the customer will leave, produced by a pre-trained Logistic Regression model.

### Risk levels
Each prediction is placed in a simple risk band so the team can act quickly:

| Risk level | Churn probability |
|------------|-------------------|
| LOW        | below 30%         |
| MEDIUM     | 30% to below 50%  |
| HIGH       | 50% or above      |

A colour-coded message accompanies each level, along with a progress bar showing the probability.

### Customer inputs
The app collects the account and service details the model uses:

- **Tenure** (months with the company)
- **Monthly charges**
- **Total charges** (estimated automatically as tenure × monthly charges, with an option to enter it manually)
- **Senior citizen**
- **Contract type** (Month-to-month, One year, Two year)
- **Internet service** (Fiber optic, DSL, No)
- **Payment method**
- **Paperless billing**
- **Tech support**
- **Online security**

### Smart input handling
- If a customer has no internet service, Tech support and Online security are automatically set to "No internet service".
- Inputs are encoded and aligned to the model's expected format automatically, so no manual preparation is needed.

### Transparency
An expandable section shows the exact encoded values sent to the model for each prediction.

### Prediction only
The app uses a model that was already trained. No training happens in the app, so results are fast and consistent.
