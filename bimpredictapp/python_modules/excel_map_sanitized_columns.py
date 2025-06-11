import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from IPython.display import display
def analyze_relationships(df_dict):
    """
    Map sanitized column names across all DataFrames after cleaning, including special handling
    for columns containing "coupés_Ids" or "coupants_Ids".

    This version expects the DataFrames to already have sanitized column names (spaces removed, etc.)

    Args:
        df_dict: Dictionary of sanitized DataFrames {'Murs': murs_df, 'Sols': sols_df, ...}

    Returns:
        Dictionary showing column mappings between DataFrames with special relationship mappings
    """
    # Get all unique column names across all DataFrames
    all_columns = set()
    df_columns = {}

    for df_name, df in df_dict.items():
        df_columns[df_name] = list(df.columns)
        all_columns.update(df.columns)

    # Find common columns across DataFrames
    common_columns = {}
    unique_columns = {}

    # Special handling for "coupés_Ids" and "coupants_Ids" columns
    relationship_mappings = {}

    # Define target columns for each DataFrame (using sanitized names)
    target_columns = {
        'Sols': 'Sol_Id',
        'Poutres': 'Poutre_Id',
        'Poteaux': 'Poteau_Id',
        'Murs': 'Mur_Id'
    }

    for column in all_columns:
        dfs_with_column = []
        for df_name, columns in df_columns.items():
            if column in columns:
                dfs_with_column.append(df_name)

        # Check if this is a relationship column (using sanitized names)
        is_relationship_col = False
        target_df = None

        if ('coupés_Ids' in column or 'coupants_Ids' in column) and len(dfs_with_column) == 1:
            # Determine which DataFrame this column is pointing to
            source_df = dfs_with_column[0]
            for potential_target, target_col in target_columns.items():
                if potential_target.lower() in column.lower() and potential_target != source_df:
                    target_df = potential_target
                    target_col_name = target_col
                    is_relationship_col = True
                    break

            if is_relationship_col:
                relationship_mappings[column] = {
                    'source_df': source_df,
                    'target_df': target_df,
                    'target_column': target_col_name,
                    'relationship_type': 'coupés/coupants'
                }

        if is_relationship_col:
            continue
        elif len(dfs_with_column) > 1:
            common_columns[column] = dfs_with_column
        else:
            if column not in unique_columns:
                unique_columns[column] = []
            unique_columns[column] = dfs_with_column[0]

    # Create mapping results
    mapping_results = {
        'common_columns': common_columns,
        'unique_columns': unique_columns,
        'relationship_columns': relationship_mappings,
        'df_column_counts': {df_name: len(columns) for df_name, columns in df_columns.items()},
        'total_unique_columns': len(all_columns)
    }

    return mapping_results






def display_column_mapping_results(column_mapping, df_dict):
    """
    Display the results of the column mapping analysis in a formatted way.

    Args:
        column_mapping: The dictionary returned by map_sanitized_columns()
        df_dict: Original dictionary of DataFrames used for the analysis
    """
    print("🔍 COLUMN MAPPING ANALYSIS")
    print("=" * 50)

    print(f"\n📊 DataFrame Column Counts:")
    for df_name, count in column_mapping['df_column_counts'].items():
        print(f"   {df_name}: {count} columns")

    print(f"\n🌐 Total Unique Columns Across All DataFrames: {column_mapping['total_unique_columns']}")

    print(f"\n🤝 Common Columns (found in multiple DataFrames): {len(column_mapping['common_columns'])}")
    for column, dfs in column_mapping['common_columns'].items():
        print(f"   '{column}' → Found in: {', '.join(dfs)}")

    print(f"\n🔗 Relationship Columns (coupés/coupants mappings): {len(column_mapping['relationship_columns'])}")
    for column, mapping in column_mapping['relationship_columns'].items():
        print(f"   '{column}' in {mapping['source_df']} → references '{mapping['target_column']}' in {mapping['target_df']}")

    print(f"\n🎯 DataFrame-Specific Columns: {len(column_mapping['unique_columns'])}")
    unique_by_df = {}
    for column, df_name in column_mapping['unique_columns'].items():
        if df_name not in unique_by_df:
            unique_by_df[df_name] = []
        unique_by_df[df_name].append(column)

    for df_name, columns in unique_by_df.items():
        print(f"\n   {df_name} unique columns ({len(columns)}):")
        for col in sorted(columns)[:10]:
            print(f"      - {col}")
        if len(columns) > 10:
            print(f"      ... and {len(columns) - 10} more")

    print(f"\n📋 COMMON COLUMNS MATRIX:")
    common_cols_df = pd.DataFrame(index=sorted(column_mapping['common_columns'].keys()),
                             columns=list(df_dict.keys()))

    for column, dfs in column_mapping['common_columns'].items():
        for df_name in df_dict.keys():
            common_cols_df.loc[column, df_name] = '✓' if df_name in dfs else '✗'

    display(common_cols_df)

def generate_relationship_graph_matplotlib(mapping_results):
    """Matplotlib version of the relationship graph"""
    plt.figure(figsize=(12, 8))

    G = nx.DiGraph()

    # Add nodes
    for df_name in mapping_results['df_column_counts'].keys():
        G.add_node(df_name, size=20)

    # Add edges for common columns
    for col, dfs in mapping_results['common_columns'].items():
        for i in range(len(dfs)-1):
            for j in range(i+1, len(dfs)):
                G.add_edge(dfs[i], dfs[j], label=col, color='gray', style='dashed')

    # Add edges for relationship columns
    for col, mapping in mapping_results['relationship_columns'].items():
        G.add_edge(
            mapping['source_df'],
            mapping['target_df'],
            label=f"{col}→{mapping['target_column']}",
            color='red',
            style='solid'
        )

    # Get positions using spring layout
    pos = nx.spring_layout(G, k=3, iterations=50)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue',
                          node_size=3000, alpha=0.8)

    # Draw edges with different styles
    # Common column edges (dashed gray)
    common_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('color') == 'gray']
    if common_edges:
        nx.draw_networkx_edges(G, pos, edgelist=common_edges,
                              edge_color='gray', style='dashed', alpha=0.6)

    # Relationship edges (solid red)
    relation_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('color') == 'red']
    if relation_edges:
        nx.draw_networkx_edges(G, pos, edgelist=relation_edges,
                              edge_color='red', style='solid', width=2)

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')

    plt.title('DataFrame Relationships', fontsize=16, fontweight='bold')
    plt.axis('off')

    # Add legend
    legend_elements = [
        Line2D([0], [0], color='gray', linestyle='--', label='Common Columns'),
        Line2D([0], [0], color='red', linestyle='-', linewidth=2, label='ID Relationships')
    ]
    plt.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.show()
