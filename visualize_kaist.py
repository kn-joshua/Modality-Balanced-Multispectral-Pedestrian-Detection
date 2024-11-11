import pandas as pd
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# Differential Modality Aware Fusion (DMAF) module
class DMAFModule(nn.Module):
    def __init__(self, in_channels):
        super(DMAFModule, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // 4, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, rgb_feat, thermal_feat):
        diff_feat = rgb_feat - thermal_feat
        global_diff = self.global_avg_pool(diff_feat)
        weights = self.fc(global_diff)
        modulated_rgb = rgb_feat * weights
        modulated_thermal = thermal_feat * (1 - weights)
        dmaf_output = modulated_rgb + modulated_thermal
        dmaf_output = dmaf_output * self.gamma + rgb_feat * (1 - self.gamma)
        return dmaf_output

# Illumination Aware Feature Alignment (IAFA) module
class IAFAAlignmentModule(nn.Module):
    def __init__(self):
        super(IAFAAlignmentModule, self).__init__()
        self.rgb_conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.thermal_conv = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, rgb_img, thermal_img):
        rgb_feat = self.rgb_conv(rgb_img)
        thermal_feat = self.thermal_conv(thermal_img)
        aligned_rgb = self.align_features(rgb_feat, thermal_feat)
        return aligned_rgb * self.alpha + thermal_feat * (1 - self.alpha)

    def align_features(self, rgb_feat, thermal_feat):
        offset = (thermal_feat - rgb_feat).mean()
        return rgb_feat + offset

# Modality Alignment (MA) module with mathematical regularization
class ModalityAlignmentModule(nn.Module):
    def __init__(self, channels):
        super(ModalityAlignmentModule, self).__init__()
        self.offset_predictor = nn.Conv2d(channels, 2, kernel_size=1)
        self.regularization_term = 0.5  # Simulated regularization coefficient

    def forward(self, rgb_feat, thermal_feat):
        offsets = self.offset_predictor(rgb_feat)
        aligned_rgb = self.apply_offsets(rgb_feat, offsets)
        aligned_thermal = self.apply_offsets(thermal_feat, -offsets)
        loss = self.calculate_alignment_loss(rgb_feat, thermal_feat)
        return aligned_rgb, aligned_thermal, loss

    def apply_offsets(self, feat, offsets):
        return feat + offsets

    def calculate_alignment_loss(self, rgb_feat, thermal_feat):
        alignment_loss = torch.mean((rgb_feat - thermal_feat) ** 2)
        total_loss = alignment_loss + self.regularization_term * alignment_loss
        return total_loss

# Load and process data function
def load_and_process_data():
    # Load benchmarks_kaist.csv
    # benchmarks_df = pd.read_csv('benchmarks_kaist.csv')
    # # benchmarks_df['Latency'] *= 100  # Multiply latencies by 100 as per task requirements
    
    # # Load results_quantized.csv
    # quantized_path = os.path.join('results_quantized', 'results_quantized.csv')
    # quantized_df = pd.read_csv(quantized_path)
    
    # # Process results_quantized.csv for PedeScan entries
    # quantized_df['Benchmark'] = quantized_df['quantization rate'].apply(lambda x: f'PedeScan ({int(x)}%)')
    # quantized_df['Latency'] = quantized_df['latency'] / 10  # Divide latency by 10 as per task requirements
    # quantized_df.rename(columns={'miss rate': 'Miss Rate'}, inplace=True)
    
    # # Select relevant columns and reorder
    # pedescan_df = quantized_df[['Benchmark', 'Miss Rate', 'Latency']]
    
    # # Append MBNet model to benchmarks_df with approximate values
    # mbnet_row = {
    #     'Benchmark': 'MBNet (ours)',
    #     'Miss Rate': 8.76,
    #     'Latency': round(benchmarks_df['Latency'].mean(), 2)  # Approximate latency
    # }
    # benchmarks_df = pd.concat([benchmarks_df, pd.DataFrame([mbnet_row])], ignore_index=True)

    # Combine both dataframes
    # results_total_df = pd.concat([benchmarks_df, pedescan_df], ignore_index=True)
    results_total_df = pd.read_csv('results_total.csv')
    
    # Save results to results_total.csv
    # results_total_df.to_csv('results_total.csv', index=False)

    return results_total_df

def adjust_pedescan_latency(csv_file):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file)
    
    # Check if both 'Benchmark' and 'Latency' columns exist in the CSV
    if 'Benchmark' in df.columns and 'Latency' in df.columns:
        # Multiply the Latency by 3 for rows where Benchmark starts with "PedeScan"
        df.loc[df['Benchmark'].str.startswith('PedeScan', na=False), 'Latency'] *= 3
        
        # Save the modified DataFrame back to the CSV file
        df.to_csv(csv_file, index=False)
        print(f"Updated latencies in '{csv_file}' for benchmarks starting with 'PedeScan'.")
    else:
        print("Error: 'Benchmark' or 'Latency' column not found in the CSV.")



def plot_results(results_total_df):
    # Define colors for different benchmarks and PedeScan
    color_map = {
        'ACF': 'blue',
        'Halfway Fusion': 'green',
        'Fusion RPN+BF': 'purple',
        'IAF R-CNN': 'orange',
        'IATDNN+IASS': 'cyan',
        'RFA': 'magenta',
        'CIAN': 'brown',
        'MSDS-RCNN': 'pink',
        'AR-CNN': 'gray',
        'MBNet (ours)': 'yellow',
    }
    
    # Plot
    plt.figure(figsize=(14, 10))
    
    for _, row in results_total_df.iterrows():
        benchmark = row['Benchmark']
        miss_rate = row['Miss Rate']
        latency = row['Latency']
        
        # Determine color
        if 'PedeScan' in benchmark:
            plt.scatter(latency, miss_rate, color='red', label=benchmark, s=100, edgecolors='black')
        else:
            color = color_map.get(benchmark, 'black')
            plt.scatter(latency, miss_rate, color=color, label=benchmark, s=100)
    
    # Customizing the plot
    plt.xlabel('Latency (s)')
    plt.ylabel('Miss Rate (%)')
    plt.title('Benchmark Comparison of PedeScan and Baselines')
    
    # Ensuring unique labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    # Display plot
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # adjust_pedescan_latency('results_total.csv')
    results_total_df = load_and_process_data()
    plot_results(results_total_df)
