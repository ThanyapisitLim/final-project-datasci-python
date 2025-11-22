import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# 1. ตรวจสอบข้อมูล
print(df.shape)
print(df['Churn'].value_counts())

# 2. อัตราการออกจากบริการ
churn_rate = (df['Churn'] == 'Yes').sum() / len(df)
print(f'อัตราการออก: {churn_rate*100:.2f}%')

# 3. เปรียบเทียบตามกลุ่ม
print(df.groupby('gender')['Churn'].apply(lambda x:(x=='Yes').sum()/len(x)))

# 4. วิเคราะห์สัญญา
contract_churn = df.groupby('Contract')['Churn'].apply(lambda x: (x=='Yes').sum()/len(x))
print(contract_churn)

# 5. กราฟแสดง
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
df['Churn'].value_counts().plot(kind='bar', ax=axes[0])
axes[0].set_title('Churn Distribution')
df.groupby('gender')['Churn'].apply(lambda x:(x=='Yes').sum()/len(x)).plot(kind='bar', ax=axes[1])
axes[1].set_title('Churn Rate by Gender')
plt.show()