import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
import networkx as nx

pio.renderers.default = "browser"

def link_id_relationships(df_dict):
    """
    Dynamically links related features between DataFrames based on referenced IDs.
    Handles type conversion for merging and provides better error handling.
    """
    linked_columns = {}

    for main_df_name, main_df in df_dict.items():
        print(f"\n🔍 Processing relationships for {main_df_name}...")

        # Find relationship columns (both cleaned and original versions)
        relationship_cols = [
            col for col in main_df.columns
            if ("_coupants_Ids" in col or "_coupés_Ids" in col) and
            ("_cleaned" in col or not any(x in col for x in ["_u", "_count"]))
        ]

        for ids_col in relationship_cols:
            # Extract target DataFrame name
            target_df_name = None
            for potential_target in df_dict.keys():
                if potential_target.lower() in ids_col.lower() and potential_target != main_df_name:
                    target_df_name = potential_target
                    break

            if not target_df_name:
                print(f"⚠️ No target found for {ids_col}, skipping")
                continue

            print(f"🔗 Linking {ids_col} in {main_df_name} → {target_df_name}")

            try:
                # Prepare source IDs - ensure they're clean strings
                exploded_df = main_df[[ids_col]].copy()
                exploded_df[ids_col] = exploded_df[ids_col].astype(str).str.replace(" ", "").str.split(',')
                exploded_df = exploded_df.explode(ids_col).dropna()
                exploded_df[ids_col] = pd.to_numeric(exploded_df[ids_col], errors='coerce').dropna()

                # Get target DataFrame and find its ID column
                target_df = df_dict[target_df_name]
                target_id_col = next(
                    (col for col in target_df.columns
                     if col.endswith("_Id") or col == "Id"),
                    None
                )

                if not target_id_col:
                    print(f"⚠️ No ID column found in {target_df_name}, skipping")
                    continue

                # Ensure target ID column is numeric
                target_df[target_id_col] = pd.to_numeric(target_df[target_id_col], errors='coerce')

                # Merge the DataFrames
                merged_df = exploded_df.merge(
                    target_df,
                    left_on=ids_col,
                    right_on=target_id_col,
                    how="left"
                )

                # Aggregate features back to original DataFrame
                feature_columns = [col for col in target_df.columns if col != target_id_col]
                for feature in feature_columns:
                    new_col_name = f"{target_df_name}_{feature}_linked"

                    if np.issubdtype(merged_df[feature].dtype, np.number):
                        main_df[new_col_name] = merged_df.groupby(merged_df.index)[feature].mean()
                    else:
                        main_df[new_col_name] = merged_df.groupby(merged_df.index)[feature].agg(
                            lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
                        )

                linked_columns[ids_col] = target_df_name
                print(f"✅ Successfully linked {len(feature_columns)} features from {target_df_name}")

            except Exception as e:
                print(f"❌ Failed to process {ids_col}: {str(e)}")
                continue

    return df_dict, linked_columns

def plot_linked_relationships(df_dict, linked_columns):
    """
    Generates a bar plot showing the number of linked relationships per column across DataFrames.

    Args:
        df_dict (dict): Dictionary of processed DataFrames.
        linked_columns (dict): Mapping of linked relationship columns to their target DataFrames.
    """
    plt.figure(figsize=(10, 6))
    plt.bar(
        linked_columns.keys(),
        [len(df_dict[target]) if target in df_dict else 0 for target in linked_columns.values()],
        color="skyblue"
    )
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Number of Linked Entries")
    plt.xlabel("Relationship Columns")
    plt.title("Linked Relationships Across DataFrames")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()




def plotly_network_graph(df_dict, linked_columns):
    """
    Creates an interactive network graph using Plotly and NetworkX.

    Args:
        df_dict (dict): Dictionary containing DataFrames.
        linked_columns (dict): Mapping of linked relationship columns to their target DataFrames.

    Returns:
        Plotly figure object.
    """
    G = nx.Graph()
    edge_x, edge_y = [], []

    # Build the graph nodes and edges dynamically
    for main_df_name, main_df in df_dict.items():
        G.add_node(main_df_name, label=main_df_name, type='dataframe')

        for ids_col, target_df_name in linked_columns.items():
            if ids_col in main_df.columns and target_df_name in df_dict:
                # Explode IDs
                exploded_df = main_df[[ids_col]].copy()
                exploded_df[ids_col] = exploded_df[ids_col].astype(str).str.split(',')
                exploded_df = exploded_df.explode(ids_col).dropna()
                exploded_df[ids_col] = pd.to_numeric(exploded_df[ids_col], errors='coerce')

                # Create edges based on relationships
                for _, row in exploded_df.iterrows():
                    id_value = row[ids_col]
                    if pd.notna(id_value):
                        target_node = f"{target_df_name} ({int(id_value)})"
                        G.add_node(target_node, label=target_node, type='record')
                        G.add_edge(main_df_name, target_node)

    # Position nodes using a spring layout
    pos = nx.spring_layout(G, seed=42)
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')

    node_x, node_y, node_text, node_color = [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(G.nodes[node]['label'])
        node_color.append('blue' if G.nodes[node]['type'] == 'dataframe' else 'orange')

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="top center",
        hoverinfo='text',
        marker=dict(showscale=False, color=node_color, size=18, line_width=2)
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=dict(text='Interactive ID Relationship Network', font=dict(size=16)),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    ))

    return fig
