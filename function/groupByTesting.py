import pandas as pd

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Group by ตาม OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, Churn
havePartner = df.groupby(['OnlineSecurity', 'OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Churn']).size()

# แปลงเป็น DataFrame
havePartner_df = havePartner.reset_index(name='count')

# เพิ่มคอลัมน์ percent จากจำนวนลูกค้าทั้งหมด
havePartner_df['percent'] = (havePartner_df['count'] / len(df)) * 100

# แสดงทุกแถว
pd.set_option('display.max_rows', None)
print(havePartner_df)
