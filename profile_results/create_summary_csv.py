import os
import re
import csv
from collections import Counter, defaultdict

def parse_log_file(file_path):
    """
    Parses a single log file and returns a Counter of COMM-EVENT pairs.
    """
    event_counter = Counter()
    # Regex to capture COMM, PID, and the full EVENT string
    line_regex = re.compile(r"^\d+\.\d+\s+([\w.-]+)\s+(\d+)\s+([\w\s]+?)\s*(?:Size=.*|$)")

    try:
        with open(file_path, 'r') as f:
            # Skip header lines
            for _ in range(2):
                next(f)
            
            for line in f:
                match = line_regex.match(line.strip())
                if match:
                    comm, _, event = match.groups()
                    event_key = f"{comm}-{event.strip()}"
                    event_counter[event_key] += 1
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
    
    return event_counter

def main():
    """
    Walks through the profile_results directory, aggregates data from all
    log files, and writes a summary CSV.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_csv_path = os.path.join(script_dir, "summary_transposed.csv")

    # This will store the data in a nested dictionary format:
    # { 'comm-event': { 'dataset-size-iteration': count, ... }, ... }
    aggregated_data = defaultdict(dict)
    
    # These sets will keep track of all unique rows and columns
    all_comm_events = set()
    all_columns = set()

    print("Scanning for log files...")

    # Walk through subdirectories to find all log files
    for root, _, files in os.walk(script_dir):
        # Skip the 'histograms' directory and the root directory itself
        if "histograms" in root or os.path.samefile(root, script_dir):
            continue
        
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                
                # --- Extract dataset, size, and iteration from path/filename ---
                try:
                    # The path relative to the script directory gives us dataset/size
                    relative_path = os.path.relpath(root, script_dir)
                    path_parts = relative_path.split(os.path.sep)
                    
                    if len(path_parts) < 2: # Expecting at least dataset/size
                        print(f"Skipping file in unexpected directory structure: {file_path}")
                        continue
                        
                    dataset = path_parts[0]
                    size = path_parts[1]
                    
                    # Iteration is the last part of the filename before .txt
                    iteration = os.path.splitext(file)[0].split('_')[-1]
                    
                    if not size.isdigit() or not iteration.isdigit():
                        continue
                        
                    col_header = f"{dataset}-{size}-{iteration}"
                    all_columns.add(col_header)
                    
                    # --- Parse the file and aggregate data ---
                    file_counts = parse_log_file(file_path)
                    for event_key, count in file_counts.items():
                        all_comm_events.add(event_key)
                        aggregated_data[event_key][col_header] = count
                        
                except IndexError:
                    print(f"Could not parse size/iteration from file: {file_path}")
                except Exception as e:
                    print(f"An unexpected error occurred for file {file_path}: {e}")

    if not aggregated_data:
        print("No data was aggregated. Exiting.")
        return

    # --- Prepare data for CSV writing ---
    # Sort rows alphabetically and columns by dataset, then numerically by size and iteration
    sorted_comm_events = sorted(list(all_comm_events))
    
    def sort_key(column_header):
        parts = column_header.split('-')
        # Last two parts are always size and iteration
        iteration = int(parts[-1])
        size = int(parts[-2])
        dataset = "-".join(parts[:-2])
        return dataset, size, iteration

    sorted_columns = sorted(list(all_columns), key=sort_key)

    # --- Filtering Step ---
    # Find COMM-EVENTs (columns) that have a non-zero value in ALL dataset-size-iterations (rows)
    common_comm_events = []
    for comm_event in sorted_comm_events:
        is_present_in_all_runs = True
        for size_iteration in sorted_columns:
            if aggregated_data[comm_event].get(size_iteration, 0) == 0:
                is_present_in_all_runs = False
                break
        if is_present_in_all_runs:
            common_comm_events.append(comm_event)

    # Define the headers for the CSV file (transposed)
    fieldnames = ['dataset-size-iteration'] + common_comm_events

    print(f"Found {len(common_comm_events)} COMM-EVENT pairs common to all {len(sorted_columns)} runs.")
    print("Writing transposed and filtered data to CSV...")
    try:
        with open(output_csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            # Iterate through each run (dataset-size-iteration) to create a row in the CSV
            for row_header in sorted_columns:
                row_data = {'dataset-size-iteration': row_header}
                for col_header in common_comm_events:
                    # Get the count for the cell, defaulting to 0
                    count = aggregated_data[col_header].get(row_header, 0)
                    row_data[col_header] = count
                writer.writerow(row_data)
        
        print(f"Successfully created transposed summary CSV: {output_csv_path}")

    except IOError as e:
        print(f"Could not write to CSV file {output_csv_path}: {e}")


if __name__ == "__main__":
    main()
