import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_aggregated_events_by_size(df, output_dir):
    """
    For each dataset, plots a bar chart showing the total sum of Major Faults
    and Disk IO events, aggregated by memory size across all iterations.
    """
    print("Generating aggregated event graphs for each dataset...")

    # --- Data Preparation ---
    # Parse the index to extract dataset, size, and iteration
    try:
        split_parts = df.index.str.rsplit('-', n=2).tolist()
        parsed_df = pd.DataFrame(split_parts, index=df.index, columns=['dataset', 'size', 'iteration'])
    except Exception as e:
        print(f"Could not parse index. Make sure it is in 'dataset-size-iteration' format. Error: {e}")
        return
        
    df = pd.concat([df, parsed_df], axis=1)
    df.dropna(subset=['size', 'iteration'], inplace=True)
    df['size'] = pd.to_numeric(df['size'])

    # --- Event Aggregation ---
    # Filter columns for Major Fault and Disk IO
    major_fault_cols = [col for col in df.columns if 'Major Fault' in col]
    disk_io_cols = [col for col in df.columns if 'Disk IO' in col]

    if not major_fault_cols and not disk_io_cols:
        print("No 'Major Fault' or 'Disk IO' columns found in the data. Skipping plot generation.")
        return

    # Sum these events to create total columns
    df['Total Major Faults'] = df[major_fault_cols].sum(axis=1)
    df['Total Disk IO'] = df[disk_io_cols].sum(axis=1)

    # Group by dataset and size, then sum the totals across all iterations
    agg_df = df.groupby(['dataset', 'size'])[['Total Major Faults', 'Total Disk IO']].sum().reset_index()

    datasets = agg_df['dataset'].unique()

    # --- Plotting ---
    for dataset in datasets:
        dataset_data = agg_df[agg_df['dataset'] == dataset]
        dataset_data = dataset_data.sort_values('size')

        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(15, 8))

        # Bar chart settings
        x = np.arange(len(dataset_data['size']))
        width = 0.35

        rects1 = ax.bar(x - width/2, dataset_data['Total Major Faults'], width, label='Total Major Faults', color='indianred')
        rects2 = ax.bar(x + width/2, dataset_data['Total Disk IO'], width, label='Total Disk IO', color='steelblue')

        # Add some text for labels, title and axes ticks
        ax.set_ylabel('Total Event Count (Sum over all iterations)')
        ax.set_title(f'Total Major Faults and Disk IO vs. Memory Size for\n{dataset}')
        ax.set_xticks(x)
        ax.set_xticklabels(dataset_data['size'])
        ax.set_xlabel('Memory Size (config)')
        ax.legend()

        ax.bar_label(rects1, padding=3, fmt='%d')
        ax.bar_label(rects2, padding=3, fmt='%d')

        fig.tight_layout()

        output_path = os.path.join(output_dir, f'aggregated_events_{dataset}.png')
        plt.savefig(output_path)
        print(f"Saved aggregated event graph to {output_path}")
        plt.close(fig)

def main():
    """
    Main function to load data and generate plots.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "summary_full_transposed.csv")
    output_dir = os.path.join(script_dir, "analysis_plots_full")

    if not os.path.exists(csv_path):
        print(f"Error: Summary file not found at {csv_path}")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Load the data using pandas, setting the first column as the index
    df = pd.read_csv(csv_path, index_col=0)

    plot_aggregated_events_by_size(df.copy(), output_dir)

if __name__ == "__main__":
    main()
