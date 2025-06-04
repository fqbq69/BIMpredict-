import pandas as pd

### ====================
### Define essential columns for each DataFrame
### ====================
ESSENTIAL_COLUMNS = {
    "Murs": [
        "Id", "011EC_Lot", "012EC_Ouvrage", "013EC_Localisation", "014EC_Mode Constructif", "Hauteur", "Epaisseur", "AI", "AS", "Sols en intersection", "Sols coupés (u)", "Sols coupés (Ids)",
        "Sols coupants (u)", "Sols coupants (Ids)", "Sol au-dessus", "Sol en-dessous", "Fenêtres", "Portes", "Ouvertures", "Murs imbriqués",
        "Mur multicouche", "Profil modifié", "Extension inférieure", "Extension supérieure", "Volume", "Surface", "Partie inférieure attachée", "Partie supérieure attachée",
        "Décalage supérieur", "Décalage inférieur", "Matériau structurel",
    ],
    "Sols": [
        "Id", "011EC_Lot", "012EC_Ouvrage", "013EC_Localisation", "014EC_Mode Constructif", "Murs en intersection",
        "Murs coupés (u)", "Murs coupés (Ids)", "Murs coupants (u)", "Murs coupants (Ids)", "Poutres en intersection", "Poutres coupés (u)",
        "Poutres coupés (Ids)", "Poutres coupants (u)", "Poutres coupants (Ids)", "Poteaux en intersection",
        "Poteaux coupés (u)", "Poteaux coupés (Ids)", "Poteaux coupants (u)", "Poteaux coupants (Ids)", "Volume", "Surface", "Matériau structurel",
    ],
    "Poutres": [
        "Id", "011EC_Lot", "012EC_Ouvrage", "013EC_Localisation", "014EC_Mode Constructif", "AI", "AS", "Hauteur totale", "Hauteur", "Sols en intersection", "Sols coupés (u)",
        "Sols coupés (Ids)", "Sols coupants (u)", "Sols coupants (Ids)", "Sol au-dessus", "Sol en-dessous", "Poteaux en intersection",
        "Poteaux coupés (u)", "Poteaux coupés (Ids)", "Poteaux coupants (u)", "Matériau structurel", "Elévation à la base", "Longueur de coupe",
    ],
    "Poteaux": [
        "Id", "011EC_Lot", "012EC_Ouvrage", "013EC_Localisation", "014EC_Mode Constructif", "Nom", "AI", "AS", "Hauteur", "Longueur",
        "Partie inférieure attachée", "Partie supérieure attachée", "Sols en intersection", "Sols coupés (u)", "Sols coupés (Ids)",
        "Sols coupants (u)", "Sols coupants (Ids)", "Poutres en intersection", "Poutres coupés (u)", "Poutres coupés (Ids)", "Poutres coupants (u)",
        "Poutres coupants (Ids)", "Matériau structurel", "Marque d'emplacement du poteau", "Décalage supérieur", "Décalage inférieur",
    ]
}

### ====================
### Data Loading and Cleaning
### ====================
def load_and_clean_data(filepath):
    """Load and clean data, then return a dictionary of DataFrames."""
    dfs = {}

    try:
        xls = pd.ExcelFile(filepath)
        available_sheets = xls.sheet_names

        for sheet, keep_cols in ESSENTIAL_COLUMNS.items():
            if sheet in available_sheets:
                df = pd.read_excel(filepath, sheet_name=sheet)
                df.columns = df.columns.str.strip().str.replace('\s+', ' ', regex=True)

                existing_cols = [col for col in keep_cols if col in df.columns]
                missing_cols = set(keep_cols) - set(existing_cols)

                if missing_cols:
                    print(f"⚠️ {sheet}: Missing {len(missing_cols)} columns: {list(missing_cols)[:3]}{'...' if len(missing_cols) > 3 else ''}")

                dfs[sheet] = df[existing_cols]
                print(f"✅ {sheet}: Kept {len(existing_cols)}/{len(keep_cols)} columns | New shape: {dfs[sheet].shape}")
            else:
                print(f"⚠️ Sheet '{sheet}' not found")
                dfs[sheet] = pd.DataFrame()

    except Exception as e:
        print(f"🚨 Error: {str(e)[:100]}...")
        dfs = {sheet: pd.DataFrame() for sheet in ESSENTIAL_COLUMNS.keys()}

    # Print cleaned DataFrame shapes
    print("\nCleaned DataFrame Shapes:")
    for name, df in dfs.items():
        print(f"{name}: {df.shape}")

    # Display Column Names
    print("\nColumn Names in Cleaned DataFrames:")
    for sheet_name, df in dfs.items():
        print(f"\nColumns in {sheet_name}:")
        print(df.columns)


    return dfs  # Returning only the dictionary of cleaned DataFrames
