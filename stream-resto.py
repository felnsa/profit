import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OrdinalEncoder

# Judul aplikasi
st.title('🌟 Prediksi Profit Restoran 🌟')

# Membaca data dari file CSV
df = pd.read_csv('restaurant_menu_optimization_data.csv')
df['Profitability Number'] = df['Profitability'].apply(lambda x: 0 if x == 'Low' else 1 if x == 'Medium' else 2)

df = df.dropna()
df = df.drop_duplicates()

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
y_pred = decision_tree.predict(X_test)

# Daftar kategori yang tersedia
categories = ['Beverages', 'Appetizers', 'Desserts', 'Main Course'] 

# Buat peta kategori ke angka
category_map = {category: idx for idx, category in enumerate(categories)}

st.write('Masukkan nilai untuk prediksi:')

# Input untuk kategori menggunakan selectbox
feature1 = st.selectbox('Menu Category', options=categories)

# Mengonversi kategori yang dipilih menjadi angka
feature1_numerik = category_map[feature1]

# Input untuk harga dengan number_input, memungkinkan desimal
feature2 = st.number_input('Price', step=0.01, format="%.2f")

# Membuat prediksi berdasarkan input user
if st.button('🔍 Prediksi'):
    # Mengonversi data input ke format yang sesuai dengan model
    features_input = [feature1_numerik, feature2]
    
    # Melakukan prediksi
    prediksi_profit = decision_tree.predict([features_input])
    st.success(f'💰 Prediksi Profit: {prediksi_profit[0]}')

    # Tampilkan hasil prediksi
    st.write(f'Prediksi Profit: {prediksi_profit[0]}')
    st.write("Catatan: Pastikan harga dimasukkan dengan format desimal seperti 2.50.")
    st.write("Catatan: Profit 0 = Low, Profit 1 = Medium, Profit 2 = High")


 
