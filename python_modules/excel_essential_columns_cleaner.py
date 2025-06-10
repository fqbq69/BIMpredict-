import pandas as pd

### ===============================================
### Define essential columns for each DataFrame
### ===============================================
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
### ========================================
### Prefix mapping for different DataFrames
### ========================================
PREFIXES = {
    "Murs": "Mur_",
    "Sols": "Sol_",
    "Poutres": "Poutre_",
    "Poteaux": "Poteau_",
}

### ============================================================================
### Function to sanitize column names by removing spaces and special characters
### ============================================================================
def sanitize_column_name(col_name):
    """Remove spaces and special characters from column names."""
    return col_name.strip().replace(" ", "_").replace("(", "").replace(")", "")


### ============================================================================
### Function to load and choose essential columns from columns in Excel sheets
### ============================================================================
def load_and_sanitize_data(filepath):
    """
    Load, clean, and sanitize all DataFrames from Excel file.
    Returns dictionary of fully sanitized DataFrames.
    """
    dfs = {}

    try:
        print("📂 Loading and sanitizing Excel file...")
        xls = pd.ExcelFile(filepath)

        for sheet, keep_cols in ESSENTIAL_COLUMNS.items():
            if sheet in xls.sheet_names:
                # Load original data
                df = pd.read_excel(filepath, sheet_name=sheet)

                # Clean and sanitize column names
                sanitized_cols = {col: sanitize_column_name(col) for col in df.columns if col in keep_cols}
                prefix = PREFIXES.get(sheet, "")
                renamed_cols = {col: prefix + sanitized_cols[col] for col in sanitized_cols}

                # Create sanitized DataFrame
                dfs[sheet] = df[list(sanitized_cols.keys())].rename(columns=renamed_cols)

                # Also sanitize the data values in ID columns
                for col in dfs[sheet].columns:
                    if col.endswith('_Id') or col.endswith('_Ids'):
                        dfs[sheet][col] = dfs[sheet][col].astype(str).str.strip()

                print(f"✅ {sheet}: Sanitized {len(renamed_cols)} columns")

    except Exception as e:
        print(f"🚨 Error: {str(e)}")
        return {sheet: pd.DataFrame() for sheet in ESSENTIAL_COLUMNS.keys()}

    return dfs
