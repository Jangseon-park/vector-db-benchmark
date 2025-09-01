import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_size_trends(df, output_dir):
    """
    For each dataset, plots a line graph showing how event counts change with memory size.
    Averages the results across iterations.
    """
    print("Generating size trend graphs for each dataset...")
    
    # Robustly parse index into components. This handles cases where the index
    # might not have 3 parts, by creating a list of lists.
    split_parts = df.index.str.rsplit('-', n=2).tolist()
    
    # Create a new DataFrame from the parsed parts. Pandas handles ragged lists
    # by filling missing values with None.
    parsed_df = pd.DataFrame(split_parts, index=df.index, columns=['dataset', 'size', 'iteration'])
    
    # Merge the parsed columns back into the original dataframe
    df = pd.concat([df, parsed_df], axis=1)

    # Crucially, drop any rows where parsing did not yield a valid size and iteration
    df.dropna(subset=['size', 'iteration'], inplace=True)
    
    df['size'] = pd.to_numeric(df['size'])
    
    # Group by dataset and size, then calculate the mean across iterations
    avg_df = df.groupby(['dataset', 'size']).mean(numeric_only=True)

    datasets = avg_df.index.get_level_values('dataset').unique()

    for dataset in datasets:
        dataset_data = avg_df.loc[dataset]
        
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(15, 8))
        
        target_comms = ['qdrant', 'search', 'general']
        target_events = ['Disk IO', 'Major Fault']

        for column in dataset_data.columns:
            if column not in ['iteration']:
                # Ensure we have a valid column format to check
                if '-' in column:
                    comm_part = column.split('-')[0]
                    event_part = column.split('-', 1)[1]

                    # Check if the process and event are in our target lists
                    is_target_comm = any(comm_part.lower().startswith(tc.lower()) for tc in target_comms)
                    is_target_event = any(te.lower() in event_part.lower() for te in target_events)

                    if is_target_comm and is_target_event:
                        ax.plot(dataset_data.index, dataset_data[column], marker='o', linestyle='-', label=column)
        
        ax.set_title(f'Event Counts vs. Memory Size for\n{dataset}', fontsize=16)
        ax.set_xlabel('Memory Size (config)', fontsize=12)
        ax.set_ylabel('Average Event Count', fontsize=12)
        ax.legend(title='COMM-EVENT', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True)
        
        plt.xticks(rotation=45)
        plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
        
        output_path = os.path.join(output_dir, f'trend_{dataset}.png')
        plt.savefig(output_path)
        print(f"Saved trend graph to {output_path}")
        plt.close(fig)

def plot_io_fault_correlation(df, output_dir):
    """
    Plots a bar chart showing the ratio of Major Faults per Disk IO for each
    main process (COMM), to show normalized I/O efficiency.
    """
    print("\nGenerating normalized Disk IO vs. Major Fault ratio graph...")
    
    # Extract COMM from column headers
    columns = df.columns
    comm_data = {}

    unique_comms = set(col.split('-')[0] for col in columns)

    target_comms = ['qdrant', 'search', 'general']
    # Filter unique_comms to only include ones that start with our targets
    unique_comms = {comm for comm in unique_comms if any(comm.lower().startswith(tc.lower()) for tc in target_comms)}

    for comm in unique_comms:
        io_col = f"{comm}-Disk IO"
        fault_col = f"{comm}-Major Fault"
        
        if io_col in columns and fault_col in columns:
            avg_io = df[io_col].mean()
            avg_fault = df[fault_col].mean()
            
            # Calculate the ratio, handle division by zero
            if avg_io > 0:
                ratio = avg_fault / avg_io
            else:
                ratio = 0
            comm_data[comm] = {'ratio_fault_per_io': ratio}
    
    if not comm_data:
        print("No COMMs with both Disk IO and Major Fault events found. Skipping correlation plot.")
        return

    # Sort by the ratio for better visualization
    sorted_comms = sorted(comm_data.items(), key=lambda item: item[1]['ratio_fault_per_io'], reverse=True)
    
    if not sorted_comms:
        print("No valid data to plot. Skipping.")
        return
        
    labels = [item[0] for item in sorted_comms]
    values = [item[1]['ratio_fault_per_io'] for item in sorted_comms]

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.bar(labels, values, color='skyblue')
    
    ax.set_title('Normalized I/O Inefficiency: Major Faults per Disk IO', fontsize=16)
    ax.set_xlabel('Process (COMM)', fontsize=12)
    ax.set_ylabel('Average Major Faults per Disk IO Event', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'correlation_ratio_fault_per_io.png')
    plt.savefig(output_path)
    print(f"Saved correlation ratio graph to {output_path}")
    plt.close(fig)

def plot_major_fault_contribution(df, output_dir):
    """
    Plots a pie chart showing the contribution of each process to the total
    number of Major Faults across all runs.
    """
    print("\nGenerating Major Fault contribution pie chart...")

    # Filter for columns that are Major Fault events
    fault_columns = [col for col in df.columns if 'Major Fault' in col]

    target_comms = ['qdrant', 'search', 'general']
    # Further filter to include only target comms
    fault_columns = [
        col for col in fault_columns
        if any(col.lower().startswith(tc.lower()) for tc in target_comms)
    ]
    
    if not fault_columns:
        print("No Major Fault columns for target comms found. Skipping contribution plot.")
        return

    # Calculate the sum of faults for each process across all runs
    fault_sums = df[fault_columns].sum()
    
    # Prettify the labels by removing the '-Major Fault' suffix
    fault_sums.index = fault_sums.index.str.replace('-Major Fault', '')

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Create the pie chart
    wedges, texts, autotexts = ax.pie(
        fault_sums, 
        labels=fault_sums.index, 
        autopct='%1.1f%%', 
        startangle=140,
        pctdistance=0.85
    )
    
    plt.setp(autotexts, size=10, weight="bold")
    ax.set_title('Contribution to Total Major Faults by Process', fontsize=16)
    
    # Equal aspect ratio ensures that pie is drawn as a circle.
    ax.axis('equal')  
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'major_fault_contribution.png')
    plt.savefig(output_path)
    print(f"Saved Major Fault contribution chart to {output_path}")
    plt.close(fig)


def main():
    """
    Main function to load data and generate plots.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "summary_full_transposed.csv")
    output_dir = os.path.join(script_dir, "analysis_plots")

    if not os.path.exists(csv_path):
        print(f"Error: Summary file not found at {csv_path}")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Load the data using pandas, setting the first column as the index
    df = pd.read_csv(csv_path, index_col=0)

    plot_size_trends(df.copy(), output_dir)
    plot_io_fault_correlation(df.copy(), output_dir)
    plot_major_fault_contribution(df.copy(), output_dir)

if __name__ == "__main__":
    main()
