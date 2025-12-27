import pandas as pd
import os

def preprocess_data(input_path):
    df = pd.read_csv(input_path)

    processed_df = df.copy()
    
    print("Preprocessing selesai secara otomatis!")
    return processed_df

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(base_dir, "MLProject", "namadataset_preprocessing", "occupancy_processed.csv")
    
    data_siap_latih = preprocess_data(input_file)
    print(data_siap_latih.head())