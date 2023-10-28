import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load your dataset
dataset = pd.read_csv('insurance.csv')

# Preprocess the dataset
dataset = pd.get_dummies(data=dataset, drop_first=True)

# Split the dataset into features and target variable
x = dataset.drop(columns='charges')
y = dataset['charges']

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Create and train the Linear Regression model
regressor_lr = LinearRegression()
regressor_lr.fit(x_train, y_train)

# Create and train the Random Forest model
regressor_rf = RandomForestRegressor(n_estimators=10, random_state=0)
regressor_rf.fit(x_train, y_train)

st.title('Insurance Cost Prediction')

st.sidebar.header('User Input')

# Create input fields for user input
age = st.sidebar.number_input('Age', min_value=1, max_value=100, value=30)
bmi = st.sidebar.number_input('BMI', min_value=10.0, max_value=50.0, value=25.0)
children = st.sidebar.number_input('Number of Children', min_value=0, max_value=10, value=0)
sex = st.sidebar.radio('Sex', ['male', 'female'])
smoker = st.sidebar.radio('Smoker', ['yes', 'no'])
region = st.sidebar.selectbox('Region', ['northeast', 'northwest', 'southeast', 'southwest'])

# Create a function to preprocess user input
def preprocess_input(age, bmi, children, sex, smoker, region):
    sex = 1 if sex == 'male' else 0
    smoker = 1 if smoker == 'yes' else 0
    region_encoded = [0, 0, 0]
    if region == 'northeast':
        region_encoded[0] = 1
    elif region == 'northwest':
        region_encoded[1] = 1
    elif region == 'southeast':
        region_encoded[2] = 1
    return [age, bmi, children, sex, smoker] + region_encoded

user_input = preprocess_input(age, bmi, children, sex, smoker, region)

# Create a function to make predictions
def predict_charges(model, user_input):
    user_input = np.array(user_input).reshape(1, -1)
    return model.predict(user_input)

# Calculate predictions for both models
lr_prediction = predict_charges(regressor_lr, user_input)
rf_prediction = predict_charges(regressor_rf, user_input)

st.sidebar.header('Prediction')

st.sidebar.write('Linear Regression Model Prediction: $', round(lr_prediction[0], 2))
st.sidebar.write('Random Forest Model Prediction: $', round(rf_prediction[0], 2))

# Create a Reset button to clear user input
if st.sidebar.button('Reset'):
    st.experimental_rerun()

# Display model comparison
st.write('Comparing Models')
st.markdown("<div>Linear Regression R^2 Score: <b>{:.2f}</b></div>".format(r2_score(y_test, regressor_lr.predict(x_test))), unsafe_allow_html=True)
st.markdown("<div>Random Forest R^2 Score: <b>{:.2f}</b></div>".format(r2_score(y_test, regressor_rf.predict(x_test))), unsafe_allow_html=True)

# Add a section to display the dataset
st.write('Insurance Dataset')
st.write(dataset)

# Add a section to display the correlation matrix heatmap
st.subheader('Correlation Matrix Heatmap')
st.write('Heatmap of the correlation matrix of the dataset.')
# Calculate the correlation matrix
correlation_matrix = dataset.corr()
# Create the heatmap using Seaborn
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.8, ax=ax)
st.pyplot(fig)  # Display the heatmap in your Streamlit app

st.subheader("Done by JAZ")
st.markdown("Jonathan Dabre - <span style='color: #FF5733;'>9529</span>", unsafe_allow_html=True)
st.markdown("Alroy Pereira - <span style='color: #3333FF;'>9631</span>", unsafe_allow_html=True)
st.markdown("Zane Falcao - <span style='color: #33FF33;'>9603</span>", unsafe_allow_html=True)
