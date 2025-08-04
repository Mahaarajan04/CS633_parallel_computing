import os
import csv

# Output CSV file name
output_csv = 'summary_big.csv'

# Only proceed if the CSV doesn't already exist
if not os.path.exists(output_csv):
    data_summary = []

    # Process each .txt file in the current directory
    for filename in os.listdir('.'):
        if filename.endswith('.txt'):
            parts = filename[:-4].split('_')
            if len(parts) < 3:
                continue  # Skip files with unexpected names

            method = parts[0]
            try:
                num_nodes = int(parts[2])
            except ValueError:
                continue  # Skip if number of nodes isn't valid

            with open(filename, 'r') as f:
                lines = f.readlines()

            # Extract every third line starting from index 2
            time_lines = []
            for i in range(2, len(lines), 3):
                try:
                    floats = list(map(float, lines[i].strip().split(',')))
                    time_lines.append(floats)
                except ValueError:
                    continue  # Skip lines with invalid float data

            # Compute averages if data exists
            if time_lines:
                time_columns = list(zip(*time_lines))
                averages = [sum(col) / len(col) for col in time_columns]
                data_summary.append([method, num_nodes, *averages])

    # Sort data by number of nodes
    data_summary.sort(key=lambda x: x[1])  # Sort by the 2nd column: number of nodes

    # Write to CSV
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['method used', 'number of nodes', 'time-1', 'time-2', 'time-3'])
        writer.writerows(data_summary)

    print(f"Created {output_csv} with data from {len(data_summary)} files.")
else:
    print(f"{output_csv} already exists. No file was overwritten.")