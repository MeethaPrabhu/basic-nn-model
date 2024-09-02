# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model 

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name:
### Register Number:
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from google.colab import auth
import gspread
from google.auth import default
import pandas as pd

auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

worksheet = gc.open('12.08.2024').sheet1

rows = worksheet.get_all_values()

df = pd.DataFrame(rows[1:], columns=rows[0])
df=df.astype({'x':'float'})
df=df.astype({'y':'float'})
df
x=df[['x']].values
y=df[['y']].values
df.head()

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=33)

import numpy as np
from sklearn.preprocessing import MinMaxScaler
mn=MinMaxScaler()
mn.fit(x_train)
x_train1=mn.transform(x_train)

ai_mind=Sequential([
    Dense(8,activation = 'relu',input_shape=[1]),
    Dense(10,activation='relu'),
    Dense(1)
])

ai_mind.compile(optimizer='rmsprop',loss='mse')
ai_mind.fit(x_train1,y_train,epochs=1000)

loss_df=pd.DataFrame(ai_mind.history.history)
loss_df.plot()

loss=ai_mind.evaluate(x_test,y_test, verbose=1)
print(f"Test loss: {loss}")

new_input=np.array([[20]],dtype=np.float32)
new_input_scaled=mn.transform(new_input)
prediction=ai_mind.predict(new_input_scaled)
print(f'Predicted Value for the input {new_input[0][0]}: {prediction[0][0]}')
```
## Dataset Information
![Screenshot 2024-09-01 213238](https://github.com/user-attachments/assets/512c8f71-75bb-43f6-9c24-cdc079dfa41c)


## OUTPUT

### Training Loss Vs Iteration Plot

![image](https://github.com/user-attachments/assets/ff7934cb-732f-4159-81ff-42907d320fce)

![image](https://github.com/user-attachments/assets/b81c778c-fbf9-4737-9b02-a6dc148ff37b)



### Test Data Root Mean Squared Error

![image](https://github.com/user-attachments/assets/0e49501e-152e-4b17-9913-f8b7fedc8366)


### New Sample Data Prediction

![image](https://github.com/user-attachments/assets/54114abf-37af-48e8-ad2e-30711c9c60ae)


## RESULT

Thus, the linear regressin network is built and implemented to predict the given input
