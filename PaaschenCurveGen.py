    import os
    import pandas as pd
    import csv
    import numpy as np
    import matplotlib.pyplot as plt
    from datetime import datetime

def read_dso_csv(file_path):
    try:
        with open(file_path, mode='r') as file:
            csv_reader = csv.reader(file)
            
            # Skip the metadata lines until we reach the header for data
            for row in csv_reader:
                if row and row[0].startswith('Time'):
                    headers = row
                    break
            
            # Read the data rows
            data = []
            for row in csv_reader:
                if row:
                    data.append(row)
                    
        # Create DataFrame
        df = pd.DataFrame(data, columns=headers)
        
        # Convert columns to appropriate data types
        df['Time (ms)'] = pd.to_numeric(df['Time (ms)'])
        
        # Convert all "CH" columns to numeric
        for col in df.columns:
            if col.startswith('CH'):
                df[col] = pd.to_numeric(df[col])
        
        return df
    
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return None
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def read_pressure_csv(file_path):
    try:
        # Read the CSV file, handling extra columns and empty rows
        pressure_data = pd.read_csv(file_path, usecols=[0, 1], skip_blank_lines=True)
        
        # Rename columns
        pressure_data.columns = ['Date/Time', 'Pressure (micron)']
        
        # Convert 'Date/Time' to datetime format
        pressure_data['Date/Time'] = pd.to_datetime(pressure_data['Date/Time'], format='%m/%d/%y %I:%M:%S %p', errors='coerce')
        pressure_data = pressure_data.dropna(subset=['Date/Time'])
        
        # Convert all dates to 24-hour format
        pressure_data['Date/Time'] = pressure_data['Date/Time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        pressure_data['Date/Time'] = pd.to_datetime(pressure_data['Date/Time'])
        
        return pressure_data
    
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return None
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def find_maxes(df, channel):
    maxes = []
    prev = 0
    av = np.mean(df[channel])
    for k in df[channel]:
        i = abs(k)
        if prev > max(4 * i, av):
            maxes.append(prev)
            prev = i
        elif prev < i:
            prev = i
    if maxes:
        maxes.pop(0)
    if(len(maxes) > 1):    
        std = np.std(maxes, ddof=1)
    else:
        std = 10
    Lbound = np.mean(maxes) - 2 * std
    Ubound = np.mean(maxes) + 2* std
    maxes = [m for m in maxes if m >= Lbound]
    maxes = [m for m in maxes if m <= Ubound]
    
    return maxes

def calculate_median_slope(df, channel):
    slopes = np.diff(df[channel]) / np.diff(df['Time (ms)'])
    median_slope = np.median(slopes)
    return median_slope

def plot_ch_vs_time(df, channel, file_name, folder_path):
    plt.figure(figsize=(10, 6))
    plt.plot(df['Time (ms)'], df[channel], label=channel)
    plt.xlabel('Time (ms)')
    plt.ylabel(channel)
    plt.title(f'{channel} vs Time (ms) for {file_name}')
    plt.legend()
    plt.grid(True)
    
    plot_file_name = os.path.splitext(file_name)[0] + f'_{channel}.png'
    plot_file_path = os.path.join(folder_path, plot_file_name)
    plt.savefig(plot_file_path)
    plt.close()

def map_pressure_to_file(pressure_data, file_name):
    # Extract timestamp from file name
    file_time_str = file_name.replace("Pokit DSO Export ", "").replace(".csv", "").replace("-", " ")
    file_time = datetime.strptime(file_time_str, '%Y %m %d %H %M %S')
    
    # Find the nearest timestamp in the pressure data
    closest_time = pressure_data.iloc[(pressure_data['Date/Time'] - file_time).abs().argsort()[:1]]
    if not closest_time.empty:
        return closest_time['Pressure (micron)'].values[0]
    else:
        return None

def process_folder(folder_path, pressure_data, max_multiplier, pressure_multiplier):
    results = []
    
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv') and "Pokit DSO Export" in file_name:
            file_path = os.path.join(folder_path, file_name)
            df = read_dso_csv(file_path)
            if df is not None:
                for col in df.columns:
                    if col.startswith('CH'):
                        plot_ch_vs_time(df, col, file_name, folder_path)
                        maxes = find_maxes(df, col)
                        maxes = [m * max_multiplier for m in maxes]
                        if maxes:
                            mean_max = np.mean(maxes)
                            std_max = np.std(maxes, ddof=1)
                            median_slope = calculate_median_slope(df, col)
                            pressure = map_pressure_to_file(pressure_data, file_name)
                            if pressure is not None:
                                pressure = float(pressure) * pressure_multiplier
                            results.append({
                                'File Name': file_name,
                                'Channel': col,
                                'Mean of Maxes': mean_max,
                                'Standard Deviation of Maxes': std_max,
                                'Median Slope': median_slope,
                                'Pressure (micron)': pressure
                            })
    
    return pd.DataFrame(results)

def plot_max_vs_pressure(all_results):
    plt.figure(figsize=(10, 6))
    colors = plt.cm.jet(np.linspace(0, 1, len(all_results)))

    for idx, (folder, df) in enumerate(all_results.items()):
        plt.scatter(df['Pressure (micron)'], df['Mean of Maxes'], color=colors[idx], label=folder)
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.xlabel('Pressure * length(micron * cm)')
    plt.ylabel('Voltage')
    plt.title('Voltage vs Pressure * Length')
    plt.legend()
    plt.grid(True)
    plt.show()

def process_multiple_folders(folder_paths, pressure_file_paths, multipliers, output_path):
    all_results = {}
    
    for folder_path, pressure_file_path, multiplier in zip(folder_paths, pressure_file_paths, multipliers):
        pressure_data = read_pressure_csv(pressure_file_path)
        if pressure_data is None:
            print(f"Pressure data file {pressure_file_path} could not be read. Skipping folder {folder_path}.")
            continue
        max_multiplier = multiplier['max_multiplier']
        pressure_multiplier = multiplier['pressure_multiplier']
        results_df = process_folder(folder_path, pressure_data, max_multiplier, pressure_multiplier)
        if not results_df.empty:
            all_results[folder_path] = results_df
    
    plot_max_vs_pressure(all_results)

    # Save the combined results
    combined_results = pd.concat(all_results.values(), ignore_index=True)
    combined_results.to_csv(output_path, index=False)

# Example usage
folder_paths = [
    r"C:\Users\ethan\Downloads\Paaschen1Magnets38.8cm99.8",
    #r"C:\Users\ethan\Downloads\Paaschen1NoMag38.3cm99.8",
    r"C:\Users\ethan\Downloads\Paaschen2Mag38.3cm99.8",
    r"C:\Users\ethan\Downloads\Paaschen2NoMag38.3cm99.8",
    r"C:\Users\ethan\Downloads\Paaschen3NoMag38.3cm99.8"

    
]
pressure_file_paths = [
    r"C:\Users\ethan\Downloads\2024-06-17-17-46-07.csv",
    #r"C:\Users\ethan\Downloads\2024-06-18-11-41-41.csv",
    r"C:\Users\ethan\Downloads\2024-06-18-17-06-11.csv",
    r"C:\Users\ethan\Downloads\2024-06-24-12-07-26.csv",
    r"C:\Users\ethan\Downloads\2024-06-24-16-54-11.csv"
    
]
multipliers = [
    #{'max_multiplier': 10.48 / 0.105, 'pressure_multiplier': 38.8},
    {'max_multiplier': 10.48 / 0.105, 'pressure_multiplier': 38.8},
    {'max_multiplier': 10.48 / 0.105, 'pressure_multiplier': 38.8},
    {'max_multiplier': 10.48 / 0.105, 'pressure_multiplier': 38.8},
    {'max_multiplier': 10.48 / 0.105, 'pressure_multiplier': 38.8}
]
output_path = r"C:\Users\ethan\Downloads\combined_summary_resultslong.csv"
process_multiple_folders(folder_paths, pressure_file_paths, multipliers, output_path)
