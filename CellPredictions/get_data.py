from pyxtal.db import database
import pandas as pd
from tqdm import tqdm
import os

def create_real_dataset(db_path="../HEM.db", output_csv="data.csv"):
    if not os.path.exists(db_path):
        print(f"Error: Database file {db_path} not found.")
        return
        
    db = database(db_path)
    codes = db.get_all_codes()[:12800]  # Limit to first 1000 entries for testing; remove slicing for full dataset
    
    print(f"Found {len(codes)} total codes in the database.")
    
    data_list = []
    
    for code in tqdm(codes, desc="Extracting data from database"):
        try:
            row = db.get_row(code)
            struc = db.get_pyxtal(code)
            
            # If there's a special site, convert to subgroup to get valid representation
            # Note: This might change the space group and cell, which exactly aligns
            # with the 1-to-many data augmentation logic you want!
            if struc.has_special_site(): 
                struc = struc.to_subgroup()
                
            cell = struc.lattice.get_para(degree=True) # returns [a, b, c, alpha, beta, gamma]
            
            data_list.append({
                'smiles': row.mol_smi,
                'hall': struc.group.hall_number,
                'zprime': struc.get_zprime()[0],
                'a': cell[0],
                'b': cell[1],
                'c': cell[2],
                'alpha': cell[3],
                'beta': cell[4],
                'gamma': cell[5],
                'volume': struc.lattice.volume
            })
        except Exception as e:
            # Skip entries that fail to parse or have invalid structures
            pass

    # Save to CSV
    df = pd.DataFrame(data_list)
    df.to_csv(output_csv, index=False)
    print(f"Successfully saved {len(df)} records to {output_csv}")

if __name__ == "__main__":
    create_real_dataset()