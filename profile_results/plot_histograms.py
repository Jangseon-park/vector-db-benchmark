import os
import re
from collections import Counter
import matplotlib.pyplot as plt

def create_histogram_from_log(file_path, output_dir, sub_dir_prefix):
    """
    Reads a single log file, counts occurrences of PID-COMM-EVENT,
    and generates a histogram.
    """
    print(f"Processing file: {file_path}")
    
    # Use a counter to store the frequency of each event combination
    event_counter = Counter()

    # Regex to capture the main parts, allowing for multi-word events like "Major Fault"
    # It captures: (COMM) (PID) (EVENT up to DETAILS)
    line_regex = re.compile(r"^\d+\.\d+\s+([\w.-]+)\s+(\d+)\s+([\w\s]+?)\s*(?:Size=.*|$)")

    try:
        with open(file_path, 'r') as f:
            # Skip the first 2 header lines
            for _ in range(2):
                next(f)
            
            for line in f:
                match = line_regex.match(line.strip())
                if match:
                    comm, pid, event = match.groups()
                    # Create the combined key for the x-axis
                    event_key = f"{pid}-{comm}-{event.strip()}"
                    event_counter[event_key] += 1
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return

    if not event_counter:
        print(f"No data parsed from {file_path}. Skipping plot.")
        return

    # --- Plotting ---
    # Sort items by count (descending) for better visualization
    sorted_items = sorted(event_counter.items(), key=lambda item: item[1], reverse=True)
    labels, values = zip(*sorted_items)

    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.5), 8)) # Dynamic width

    ax.bar(labels, values)
    
    # Formatting
    ax.set_title(f'Event Histogram for\n{os.path.basename(file_path)}', fontsize=16)
    ax.set_xlabel('PID-COMM-EVENT', fontsize=12)
    ax.set_ylabel('Accumulated Call Number', fontsize=12)
    
    plt.xticks(rotation=90, ha='center')
    plt.tight_layout() # Adjust layout to make room for labels

    # --- Saving the plot ---
    base_filename = os.path.splitext(os.path.basename(file_path))[0]
    # Prepend the sub-directory name to the output filename to ensure uniqueness
    if sub_dir_prefix and sub_dir_prefix != '.':
        output_filename = f"{sub_dir_prefix}_{base_filename}.png"
    else:
        output_filename = f"{base_filename}.png"
    output_path = os.path.join(output_dir, output_filename)
    
    try:
        plt.savefig(output_path)
        print(f"Histogram saved to: {output_path}")
    except Exception as e:
        print(f"Error saving plot to {output_path}: {e}")
    finally:
        plt.close(fig) # Close the figure to free memory

def main():
    """
    Finds all log files in subdirectories and generates histograms for them.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    histograms_dir = os.path.join(script_dir, "histograms")

    if not os.path.exists(histograms_dir):
        os.makedirs(histograms_dir)
        print(f"Created output directory: {histograms_dir}")

    # Walk through all subdirectories of the current directory
    for root, _, files in os.walk(script_dir):
        if root == histograms_dir: # Don't process the output directory
            continue

        sub_dir_prefix = os.path.relpath(root, script_dir)
            
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                create_histogram_from_log(file_path, histograms_dir, sub_dir_prefix)

if __name__ == "__main__":
    print("Starting histogram generation...")
    main()
    print("All files processed.")
