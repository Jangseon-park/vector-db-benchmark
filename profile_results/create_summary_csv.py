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
    output_csv_path = os.path.join(script_dir, "summary.csv")

    # This will store the data in a nested dictionary format:
    # { 'comm-event': { 'size-iteration': count, ... }, ... }
    aggregated_data = defaultdict(dict)
    
    # These sets will keep track of all unique rows and columns
    all_comm_events = set()
    all_size_iterations = set()

    print("Scanning for log files...")

    # Walk through subdirectories to find all log files
    for root, _, files in os.walk(script_dir):
        # Skip the 'histograms' directory and the root directory itself
        if "histograms" in root or os.path.samefile(root, script_dir):
            continue
        
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                
                # --- Extract size and iteration from path/filename ---
                try:
                    # Size is the name of the parent directory
                    size = os.path.basename(root)
                    # Iteration is the last part of the filename before .txt
                    iteration = os.path.splitext(file)[0].split('_')[-1]
                    
                    if not size.isdigit() or not iteration.isdigit():
                        continue
                        
                    col_header = f"{size}-{iteration}"
                    all_size_iterations.add(col_header)
                    
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
    # Sort rows alphabetically and columns numerically by size, then iteration
    sorted_comm_events = sorted(list(all_comm_events))
    sorted_size_iterations = sorted(list(all_size_iterations), 
                                    key=lambda x: tuple(int(i) for i in x.split('-')))

    # Define the headers for the CSV file
    fieldnames = ['COMM-EVENT'] + sorted_size_iterations

    print("Writing data to CSV...")
    try:
        with open(output_csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            # This list will hold only the rows that pass the filter
            filtered_rows = []

            for event_key in sorted_comm_events:
                row = {'COMM-EVENT': event_key}
                is_common_to_all = True  # Assume it's common until proven otherwise
                
                # First, build the full row and check if it's common everywhere
                for col_header in sorted_size_iterations:
                    count = aggregated_data[event_key].get(col_header, 0)
                    row[col_header] = count
                    if count == 0:
                        is_common_to_all = False
                
                if is_common_to_all:
                    filtered_rows.append(row)
            
            writer.writerows(filtered_rows)
        
        print(f"Successfully created summary CSV: {output_csv_path}")
        print(f"Total common events found: {len(filtered_rows)}")

    except IOError as e:
        print(f"Could not write to CSV file {output_csv_path}: {e}")


if __name__ == "__main__":
    main()
