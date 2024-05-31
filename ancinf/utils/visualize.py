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
    for dataset_name, experiments  in data["details"].items():        
        for experiment in experiments:
            classifiers = experiment["classifiers"]            
            classifier_names = []
            means = []
            std_devs = []


            for name, metrics in classifiers.items():
                if name == "exp_idx":
                    continue
                classifier_names.append(name)
                means.append(metrics['f1_macro']['mean'])
                std_devs.append(metrics['f1_macro']['std'])

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
            plt.figure(figsize=(14, 6))
            sns.set(style="whitegrid")
            
            cols = []
            color_model_scheme = {'GNN_graph_based':'red', 'GNN_one_hot':'#69b3a2', 'MLP':'orange', 'Heuristics':'blue', 'community_detection': 'pink'}
            for model_name in df.Classifier:
                if 'gb' in model_name:
                    cols.append(color_model_scheme['GNN_graph_based'])
                elif 'MLP' in model_name:
                    cols.append(color_model_scheme['MLP'])
                elif model_name in ["EdgeCount", "EdgeCountPerClassize", "SegmentCount", "LongestIbd", "IbdSum", "IbdSumPerEdge"]:
                    cols.append(color_model_scheme['Heuristics'])
                elif model_name in ["Spectral", "Agglomerative", "GirvanNewmann", "LabelPropagation", "RelationalNeighbor", "MultiRankWalk", "RidgeRegression"]:
                    cols.append(color_model_scheme['community_detection'])
                else:
                    cols.append(color_model_scheme['GNN_one_hot'])

            bar_plot = sns.barplot(x='Classifier', y='Mean', data=df, ci=None, palette=cols)  # , palette="viridis")

            # Adding error bars
            for i, (mean, std) in enumerate(zip(df['Mean'], df['StdDev'])):
                bar_plot.errorbar(i, mean, yerr=std, fmt='none', c='black', capsize=5)

                # Optionally annotate the bars with the exact mean values
                if annotate:
                    bar_plot.text(i, mean + std + 0.01, f'{mean:.2f}', ha='center', va='bottom', fontsize=6)
                    
            
            for k, v in color_model_scheme.items():
                plt.scatter([],[], c=v, label=k)

            plt.title(f'Classifier Performance for {dataset_name}')
            plt.xlabel('Classifier')
            plt.ylabel('Mean Performance')
            plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
            plt.legend()
            plt.tight_layout()  # Adjust the layout to make room for the rotated labels
            plt.show()
