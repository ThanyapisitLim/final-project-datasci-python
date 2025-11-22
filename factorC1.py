import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from function.dataLoader import load_telco_data
import matplotlib.pyplot as plt
import matplotlib

plt.rcParams['font.family'] = 'Tahoma'
# ==========================
# 🔧 Fix ฟอนต์ภาษาไทย (macOS)
# ==========================
plt.rcParams['font.family'] = 'Arial Unicode MS'
plt.rcParams['axes.unicode_minus'] = False
# ถ้า Windows ใช้: matplotlib.rcParams['font.family'] = 'Tahoma'
matplotlib.rcParams['axes.unicode_minus'] = False

# *************** เพิ่มส่วนนี้เพื่อแสดงผลได้ครบถ้วน ***************
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)
# *****************************************************************

# โหลด data
df = load_telco_data("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# --- สร้าง cluster ใหม่ ---
features = ['tenure', 'MonthlyCharges', 'TotalCharges']
X = df[features].copy()

# แก้ TotalCharges null
X['TotalCharges'] = pd.to_numeric(X['TotalCharges'], errors='coerce')
X = X.fillna(0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# เพิ่ม n_init=10 เพื่อหลีกเลี่ยง Warning และเพิ่มความเสถียร
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# --- filter cluster 1 ---
cluster_1 = df[df['Cluster'] == 1].copy()

print("==================================================")
print("             📌 การวิเคราะห์ Cluster 1             ")
print("==================================================")
print("\n📌 ขนาดกลุ่ม 1:", len(cluster_1))

# อัตราการออก
churn_rate = (cluster_1['Churn'] == 'Yes').mean() * 100
print(f"🔥 อัตราการออกของ Cluster 1: {churn_rate:.2f}%")

# --- การกระจายตัวของ Contract ---
contract_distribution = cluster_1['Contract'].value_counts()
contract_proportion = (contract_distribution / len(cluster_1)) * 100

print("\n--- 📊 การกระจายตัวของ Contract ใน Cluster 1.0 ---")
print(f"จำนวนลูกค้าทั้งหมดในกลุ่ม 1: {len(cluster_1)} ราย")

contract_summary = pd.DataFrame({
    'จำนวนลูกค้า': contract_distribution,
    'สัดส่วน (%)': contract_proportion.round(2)
})
print(contract_summary.sort_values(by='จำนวนลูกค้า', ascending=False))

most_common_contract = contract_distribution.idxmax()
print(f"\n✅ Contract ที่ใช้มากที่สุดใน Cluster 1 คือ: **{most_common_contract}**")


# --- สัดส่วนการใช้ Service ---
factors_service_proportion = [
    'InternetService', 'OnlineSecurity', 'TechSupport',
    'DeviceProtection', 'StreamingTV', 'StreamingMovies',
    'PaymentMethod', 'PaperlessBilling'
]

print("\n--- 📊 สัดส่วนการใช้บริการใน Cluster 1.0 (เปอร์เซ็นต์) ---")
for f in factors_service_proportion:
    service_proportion = (cluster_1[f].value_counts() / len(cluster_1)) * 100
    proportion_summary = pd.DataFrame({'สัดส่วน (%)': service_proportion.round(2)})

    print(f"\n=== {f} (สัดส่วน) ===")
    print(proportion_summary)

# --- วิเคราะห์ Demographics ---
demographic_factors = ['gender', 'Partner', 'Dependents']

print("\n--- 🧑‍🤝‍👩 การกระจายตัวของ Demographics (Cluster 1) ---")
for f in demographic_factors:
    demographic_proportion = (cluster_1[f].value_counts(normalize=True) * 100).round(2)
    demographic_summary = pd.DataFrame({'สัดส่วน (%)': demographic_proportion})

    print(f"\n=== {f} (สัดส่วน) ===")
    print(demographic_summary)
# --- ปัจจัยที่สัมพันธ์กับ churn ---
factors_churn = [
    'Contract', 'InternetService', 'OnlineSecurity', 'TechSupport',
    'DeviceProtection', 'StreamingTV', 'StreamingMovies',
    'PaymentMethod', 'PaperlessBilling'
]

print("\n--- 📈 ปัจจัยที่สัมพันธ์กับ churn (เฉพาะ cluster 1) ---")
for f in factors_churn:
    pivot = pd.crosstab(cluster_1[f], cluster_1['Churn'], normalize='index') * 100
    print(f"\n=== {f} (Churn Rate) ===")
    print(pivot)


# ======================================
# 📊 Visualization Yes/No (Stacked Bars)
# ======================================
viz_factors = [
    'Contract', 'InternetService', 'OnlineSecurity', 'TechSupport',
    'DeviceProtection', 'StreamingTV', 'StreamingMovies',
    'PaymentMethod', 'PaperlessBilling'
]

for col in viz_factors:
    pivot = pd.crosstab(cluster_1[col], cluster_1['Churn'], normalize='index') * 100
    pivot.plot(kind='bar', stacked=True)

    plt.title(f"Churn Distribution by {col} (Cluster 1)")
    plt.xlabel(col)
    plt.ylabel("เปอร์เซ็นต์ (%)")
    plt.legend(title="Churn")
    plt.tight_layout()
    plt.show()

# ======================================
# 📊 Summary Plot for Cluster 1
# ======================================
import math

summary_factors = [
    'Contract', 'InternetService', 'OnlineSecurity', 'TechSupport',
    'DeviceProtection', 'StreamingTV', 'StreamingMovies',
    'PaymentMethod', 'PaperlessBilling'
]

num_factors = len(summary_factors)
cols = 3  # กำหนดจำนวน column ใน subplot
rows = math.ceil(num_factors / cols)

fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*4))
axes = axes.flatten()

for i, col in enumerate(summary_factors):
    pivot = pd.crosstab(cluster_1[col], cluster_1['Churn'], normalize='index') * 100
    pivot.plot(kind='bar', stacked=True, ax=axes[i])
    axes[i].set_title(f"{col} (Churn%)")
    axes[i].set_ylabel("เปอร์เซ็นต์ (%)")
    axes[i].legend(title='Churn')

# กำจัด subplot ว่าง
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.suptitle("📊 Summary of Cluster 1 Factors", fontsize=16, y=1.02)
plt.show()
