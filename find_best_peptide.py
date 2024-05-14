import csv
import pandas as pd 
import os
import sys


def find_best_peptide(original_sequence):
        

    input_directory = f"/storage/plzen4-ntis/home/jrx99/esm_mutations/{original_sequence}"
    csv_file_path = os.path.join(input_directory, f'{original_sequence}_mutation.csv')
    df=find_best_peptide(csv_file_path)
    
    column_names = ['PAE_MEAN','MAXIMUM_OF_MIN_DISTANCES','MEAN_OF_MIN_DISTANCES','proximity_same_charge','deformation','peptide'] 

    # Load the CSV file into a pandas DataFrame with specified column names
    df = pd.read_csv(csv_file_path, names=column_names)
    
    # Initial number of rows in the DataFrame
    initial_rows = len(df)

    # Thresholds
    deformation_threshold = 10
    pae_mean_threshold = 17



    df_without_u = df[~df['peptide'].str.contains('U')]
    
    
    # Filter 1: Deformation higher than threshold
    filtered_df_1 = df_without_u[df_without_u['deformation'] < deformation_threshold]
    filtered_rows_1 = len(df_without_u) - len(filtered_df_1)
    print(f"Filtered out {filtered_rows_1} rows based on deformation threshold.")

    # Filter 2: PAE mean lower than threshold
    filtered_df_2 = filtered_df_1[filtered_df_1['PAE_MEAN'] < pae_mean_threshold]
    filtered_rows_2 = len(filtered_df_1) - len(filtered_df_2)
    print(f"Filtered out {filtered_rows_2} rows based on PAE mean threshold.")

    # Select top 10 based on Criterion 1 and 2
    filtered_df_3 = filtered_df_2.nsmallest(5, ['MAXIMUM_OF_MIN_DISTANCES', 'MEAN_OF_MIN_DISTANCES'])

    # Final number of rows in the DataFrame
    final_rows = len(filtered_df_3)
    print(f"Remaining {final_rows} rows after all filters.")

    # Print the final DataFrame
    return filtered_df_3["peptide"].to_list




   
