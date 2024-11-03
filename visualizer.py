import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    # Path to the CSV file
    csv_file = os.path.join('results_quantized', 'results_quantized.csv')

    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Get the unique datasets
    datasets = df['dataset'].unique()

    for dataset in datasets:
        # Filter data for the current dataset
        df_dataset = df[df['dataset'] == dataset]

        # Sort by quantization rate
        df_dataset = df_dataset.sort_values(by='quantization rate')

        # Plot
        fig, ax1 = plt.subplots(figsize=(10, 6))

        quantization_rates = df_dataset['quantization rate']
        miss_rates = df_dataset['miss rate']
        latencies = df_dataset['latency']

        color1 = 'red'
        ax1.set_xlabel('Quantization Rate (%)')
        ax1.set_ylabel('Miss Rate (%)', color=color1)
        ax1.plot(quantization_rates, miss_rates, color=color1, label='Miss Rate')
        ax1.tick_params(axis='y', labelcolor=color1)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color2 = 'blue'
        ax2.set_ylabel('Latency (seconds)', color=color2)
        ax2.plot(quantization_rates, latencies, color=color2, label='Latency')
        ax2.tick_params(axis='y', labelcolor=color2)

        # Add other hyperparameters in the upper left corner
        epochs = df_dataset['epochs'].iloc[0]
        fppi = df_dataset['fppi'].iloc[0]
        hyperparams_text = f'Number of Epochs: {epochs}\nFPPI: {fppi}'
        plt.text(0.05, 0.95, hyperparams_text, transform=ax1.transAxes, verticalalignment='top')

        # Add title
        plt.title(f'{dataset.upper()} Dataset - Miss Rate and Latency vs Quantization Rate')

        # Add grid
        ax1.grid(True)

        # Add legends
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='lower right')

        # Save the figure
        plt.savefig(f'{dataset}_miss_rate_latency_plot.png')
        plt.show()

if __name__ == '__main__':
    main()
