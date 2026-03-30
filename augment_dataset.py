import pandas as pd
import numpy as np

def augment_data(input_csv='Crop_recommendation.csv', output_csv='Crop_recommendation.csv', new_rows_per_class=20):
    df = pd.read_csv(input_csv)
    new_rows = []
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    for label in df['label'].unique():
        class_data = df[df['label'] == label]
        
        # We will generate `new_rows_per_class` rows for this label
        for _ in range(new_rows_per_class):
            new_row = {}
            for col in df.columns:
                if col == 'label':
                    new_row[col] = label
                else:
                    # Generate a random value based on the mean and std of this class for this column
                    mean = class_data[col].mean()
                    std = class_data[col].std()
                    val = np.random.normal(mean, std)
                    
                    # Clamp the value between min and max to ensure it remains realistic
                    c_min = class_data[col].min()
                    c_max = class_data[col].max()
                    val = max(min(val, c_max), c_min)
                    
                    # If the original data appears to be integers (like N, P, K), round it
                    if class_data[col].dtype == 'int64' or col in ['N', 'P', 'K']:
                        val = int(round(val))
                    
                    new_row[col] = val
                    
            new_rows.append(new_row)
            
    df_new = pd.DataFrame(new_rows)
    df_augmented = pd.concat([df, df_new], ignore_index=True)
    
    # Save the augmented dataset
    df_augmented.to_csv(output_csv, index=False)
    
    print(f"Original shape: {df.shape}")
    print(f"Augmented shape: {df_augmented.shape}")
    print("Dataset augmentation complete.")

if __name__ == '__main__':
    augment_data()
