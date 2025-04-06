import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\lenovo\Documents\python\house_prices.csv")
df['Carpet Area'] = df['Carpet Area'].str.extract(r'(\d+)').astype(float)
df['Bathroom'] = pd.to_numeric(df['Bathroom'], errors='coerce')
df['Balcony'] = pd.to_numeric(df['Balcony'], errors='coerce')
df['Price (in rupees)'] = pd.to_numeric(df['Price (in rupees)'], errors='coerce')
df['Carpet Area'].fillna(df['Carpet Area'].mean(), inplace=True)
df['Bathroom'].fillna(0, inplace=True)
df['Balcony'].fillna(0, inplace=True)
df.dropna(subset=['Price (in rupees)'], inplace=True)
X = df[['Carpet Area', 'Bathroom', 'Balcony']]
y = df['Price (in rupees)']

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model=LinearRegression()
model.fit(x_train,y_train)

y_pred=model.predict(x_test)

print("mean:",mean_squared_error(y_test,y_pred))
print("r2:",r2_score(y_test,y_pred))

y_pred = model.predict(x_test)
# Add predicted prices to x_test for visualization
x_test_copy = x_test.copy()
x_test_copy['Predicted Price'] = y_pred

# Plot: Carpet Area vs Predicted Price
plt.figure(figsize=(8, 5))
plt.scatter(x_test_copy['Carpet Area'], x_test_copy['Predicted Price'], color='green')
plt.xlabel("Carpet Area")
plt.ylabel("Predicted Price")
plt.title("Carpet Area vs Predicted Price")
plt.grid(True)
plt.show()

# Plot: Bathroom vs Predicted Price
plt.figure(figsize=(8, 5))
plt.scatter(x_test_copy['Bathroom'], x_test_copy['Predicted Price'], color='blue')
plt.xlabel("Bathroom")
plt.ylabel("Predicted Price")
plt.title("Bathroom vs Predicted Price")
plt.grid(True)
plt.show()

# Plot: Balcony vs Predicted Price
plt.figure(figsize=(8, 5))
plt.scatter(x_test_copy['Balcony'], x_test_copy['Predicted Price'], color='orange')
plt.xlabel("Balcony")
plt.ylabel("Predicted Price")
plt.title("Balcony vs Predicted Price")
plt.grid(True)
plt.show()