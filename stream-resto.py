from pyexpat import model
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder 

# Judul aplikasi
st.title('Aplikasi Prediksi Profit')
# Membaca data dari file CSV
df = pd.read_csv('restaurant_menu_optimization_data.csv')
df['Profitability Number'] = df['Profitability'].apply(lambda x: 0 if x == 'Low' else 1 if x == 'Medium' else 2)

df=df.dropna()
df=df.drop_duplicates()

# Memilih fitur dan target
features = ['MenuCategory', 'Price']

X = df[features]
y = df['Profitability Number']
# Apply OrdinalEncoder to categorical columns in the entire dataset X BEFORE splitting
categ_cols = X.select_dtypes(include=['object']).columns
ordinal_encoder = OrdinalEncoder()
X[categ_cols] = ordinal_encoder.fit_transform(X[categ_cols])

# Membagi data menjadi training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Inisialisasi model Decision Tree
decision_tree = DecisionTreeClassifier()
# Latih model Decision Tree pada data pelatihan
decision_tree.fit(X_train, y_train)

# Membuat input untuk prediksi user
st.write('Masukkan nilai untuk prediksi:')
feature1 = st.number_input('Menu Category')
feature2 = st.number_input('Price')

# Membuat prediksi berdasarkan input user
if st.button('Prediksi'):
    user_input = [[feature1, feature2,]]
    prediksi_profit = decision_tree(user_input)
    st.write(f'Prediksi Profit: {prediksi_profit[0]}')
