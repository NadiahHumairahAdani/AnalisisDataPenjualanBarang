import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 1. Pengumpulan Data
data = pd.read_csv('data_penjualan.csv')

# 2. Data Cleaning
print("Missing values per column:\n", data.isnull().sum())
data = data.drop_duplicates()

# Memeriksa data setelah cleaning
print("Data after cleaning:\n", data.head())

# 3. Data Transformation
data['Total_Penjualan'] = data['Jumlah'] * data['Harga_Satuan']
data['Tanggal'] = pd.to_datetime(data['Tanggal'], format='%d/%m/%Y')

# Memeriksa data setelah transformasi
print("Data after transformation:\n", data.head())

# 4. Exploratory Data Analysis (EDA)
# Distribusi penjualan berdasarkan jenis kelamin
plt.figure(figsize=(8, 6))
sns.countplot(data=data, x='Jenis_Kelamin')
plt.title('Distribusi Penjualan Berdasarkan Jenis Kelamin')
plt.show()

# Penjualan per jenis barang
penjualan_per_jenis = data.groupby('Jenis_Barang')['Total_Penjualan'].sum().sort_values()
penjualan_per_jenis.plot(kind='bar', figsize=(10, 6))
plt.title('Penjualan per Jenis Barang')
plt.ylabel('Total Penjualan')
plt.xlabel('Jenis Barang')
plt.show()

# Tren penjualan bulanan
data['Bulan'] = data['Tanggal'].dt.to_period('M')
tren_bulanan = data.groupby('Bulan')['Total_Penjualan'].sum()
tren_bulanan.plot(kind='line', marker='o', figsize=(10, 6))
plt.title('Tren Penjualan Bulanan')
plt.ylabel('Total Penjualan')
plt.xlabel('Bulan')
plt.show()

# 5. Modelling Data
# Memprediksi penjualan di bulan berikutnya
data['Bulan'] = data['Tanggal'].dt.month
data['Tahun'] = data['Tanggal'].dt.year

# Menggunakan data bulan dan tahun untuk prediksi
X = data[['Bulan', 'Tahun']]
y = data['Total_Penjualan']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# 6. Validasi dan Tuning Model
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'RMSE: {rmse}')

# 7. Interpretasi dan Penyajian Hasil
plt.scatter(y_test, y_pred)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted')
plt.show()

# 8. Deploy dan Monitoring
# Proses deploy akan bergantung pada platform yang digunakan
# Monitoring bisa dilakukan dengan menyimpan prediksi ke database dan memeriksa performa model secara berkala

# 9. Maintenance dan Iterasi
# Update model dengan data baru dan lakukan iterasi sesuai kebutuhan
