from sklearn.ensemble import RandomForestRegressor
import streamlit as st
import pickle
from sklearn.preprocessing import StandardScaler
import pandas as pd
import time
from sklearn.datasets import fetch_california_housing
st.title('🏠House Price prediction using ML')

st.image('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRvQdqIasHkDTf5733FK14z5mPQ18VPhg_R_Q&s')

df = pd.read_csv('house_data.csv')
X = df.iloc[:,:-3]
y = df.iloc[:,-1]

final_X = X
scaler = StandardScaler()
scaled_X = scaler.fit_transform(final_X)

st.sidebar.title('Select House features: ')
st.sidebar.image('https://cdn.dribbble.com/userupload/20000742/file/original-aaf23458355a156d0cf85b8217a5065a.gif')
all_value = []
for i in final_X:
  min_value = final_X[i].min()
  max_value = final_X[i].max()
  result = st.sidebar.slider(f'Select {i} value',min_value,max_value)
  all_value.append(result)


user_X = scaler.transform([all_value])

@st.cache_data
def ml_model(X,y):
  model = RandomForestRegressor()
  model.fit(X,y)
  return model

model = ml_model(scaled_X,y)
house_price = model.predict(user_X)[0]

final_price = round(house_price * 100000,2)

with st.spinner('Predicting House Price'):
  import time
  time.sleep(2)

st.success(f'Estimated House Price is: $ {final_price}')
st.markdown('''**Design and Developed by: SACHIN shukla**''')
