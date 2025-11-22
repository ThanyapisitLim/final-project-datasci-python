import pandas as pd

def load_telco_data(path: str):
    """โหลดและ clean dataset"""
    df = pd.read_csv(path)
    
    # แปลง TotalCharges ให้เป็นตัวเลข (บางแถวเป็น string)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna(subset=['TotalCharges'])
    
    return df