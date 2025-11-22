import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def rule_based_segmentation(df: pd.DataFrame) -> pd.DataFrame:
    """เพิ่มคอลัมน์ Segment แบบใช้ rule-based"""
    def segment(row):
        if row['Churn'] == 'Yes' and row['Contract'] == 'Month-to-month' and row['tenure'] < 12:
            return 'High Risk'
        elif row['Churn'] == 'No' and row['tenure'] >= 24 and row['Contract'] == 'Two year':
            return 'Loyal Customer'
        elif row['MonthlyCharges'] > 90:
            return 'High Value'
        else:
            return 'Regular'

    df['Segment'] = df.apply(segment, axis=1)
    return df


def kmeans_segmentation(df: pd.DataFrame, n_clusters=4) -> pd.DataFrame:
    """เพิ่มคอลัมน์ Cluster แบบใช้ KMeans"""
    features = df[['tenure', 'MonthlyCharges', 'TotalCharges']].copy()
    features = features.apply(pd.to_numeric, errors='coerce').dropna()

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df.loc[features.index, 'Cluster'] = kmeans.fit_predict(scaled)

    return df
