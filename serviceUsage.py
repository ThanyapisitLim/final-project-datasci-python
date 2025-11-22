import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from function.dataLoader import load_telco_data

df = load_telco_data("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# --- สร้าง cluster ---
features = ['tenure', 'MonthlyCharges', 'TotalCharges']
X = df[features].copy()
X['TotalCharges'] = pd.to_numeric(X['TotalCharges'], errors='coerce')
X = X.fillna(0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# --- Churn rate ของแต่ละ cluster ---
churn_rate = df.groupby('Cluster')['Churn'].apply(lambda x: (x=='Yes').mean() * 100)
print("\n🔥 Churn rate by cluster (%)")
print(churn_rate)

# --- Service usage ---
services = ['OnlineSecurity', 'TechSupport', 'DeviceProtection', 'StreamingTV', 'StreamingMovies']

# สร้าง pivot table ของแต่ละ cluster กับ service usage
service_usage = df.groupby('Cluster')[services].apply(lambda x: (x=='Yes').mean() * 100)
print("\n📊 Service usage (%) by cluster")
print(service_usage)

# --- Heatmap visual ของ service usage ---
plt.figure(figsize=(10,5))
sns.heatmap(service_usage, annot=True, fmt=".1f", cmap="Blues")
plt.title("Service Usage (%) by Cluster")
plt.ylabel("Cluster")
plt.show()
