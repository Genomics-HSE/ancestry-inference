import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings

# Suppress all warnings
warnings.filterwarnings('ignore')

def load_data_from_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def visualize_classifier_data(data, sort_bars=False, annotate=False):
    for dataset_name, classifiers in data.items():
        classifier_names = []
        means = []
        std_devs = []

        for classifier in classifiers:
            for name, metrics in classifier.items():
                if name == "exp_idx":
                    continue
                classifier_names.append(name)
                means.append(metrics['mean'])
                std_devs.append(metrics['std'])

        # Create a DataFrame for easier plotting with seaborn
        df = pd.DataFrame({
            'Classifier': classifier_names,
            'Mean': means,
            'StdDev': std_devs
        })

        # Optionally sort the bars by their mean values
        if sort_bars:
            df = df.sort_values('Mean', ascending=False)

        # Plotting
        plt.figure(figsize=(10, 6))
        sns.set(style="whitegrid")

        bar_plot = sns.barplot(x='Classifier', y='Mean', data=df, ci=None)  # , palette="viridis")

        # Adding error bars
        for i, (mean, std) in enumerate(zip(df['Mean'], df['StdDev'])):
            bar_plot.errorbar(i, mean, yerr=std, fmt='none', c='black', capsize=5)

            # Optionally annotate the bars with the exact mean values
            if annotate:
                bar_plot.text(i, mean + std + 0.01, f'{mean:.2f}', ha='center', va='bottom', fontsize=8)

        plt.title(f'Classifier Performance for {dataset_name}')
        plt.xlabel('Classifier')
        plt.ylabel('Mean Performance')
        plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
        plt.tight_layout()  # Adjust the layout to make room for the rotated labels
        plt.show()
