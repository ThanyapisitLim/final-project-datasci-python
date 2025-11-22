import seaborn as sns
import matplotlib.pyplot as plt
from function.dataLoader import load_telco_data
from function.segmentation import rule_based_segmentation, kmeans_segmentation

# โหลดข้อมูล
df = load_telco_data("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# ทำ segmentation แบบ rule-based
df = rule_based_segmentation(df)
print("🔹 Rule-based segmentation result:")
print(df['Segment'].value_counts(normalize=True) * 100)
# 🔹 Rule-based segmentation %
rule_percent = df['Segment'].value_counts(normalize=True) * 100
print("\n📊 Rule-based segmentation (%):")
print(rule_percent.round(2))


# ทำ segmentation แบบ KMeans
df = kmeans_segmentation(df)
print("\n🔹 Cluster averages:")
print(df.groupby('Cluster')[['tenure','MonthlyCharges','TotalCharges']].mean())
# 🔹 KMeans clustering %
cluster_percent = df['Cluster'].value_counts(normalize=True) * 100
print("\n🤖 K-Means segmentation (%):")
print(cluster_percent.round(2))

# Visualization
sns.scatterplot(data=df, x='tenure', y='MonthlyCharges', hue='Cluster', palette='Set2')
plt.title("Customer Segmentation by K-Means")
plt.show()