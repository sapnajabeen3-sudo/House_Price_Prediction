import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# 1. Data load karo
df = pd.read_csv('house_prices.csv') 
X = df.drop('price', axis=1) 
y = df['price']

# 2. Train karo - Sirf Sklearn
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 3. Save karo isi naam se
joblib.dump(model, 'best_model.joblib') 
print("Model ban gaya: best_model.joblib")