import os
import re
import hashlib
import time
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm

#########################################################
################## Building Functions ###################
#########################################################

def load_cities(parent_path):
    return {
        folder_name: {
            'texts': version_texts,
            'words': {k: text_to_words(text) for k, text in version_texts.items()}
        }
        for folder_name in os.listdir(parent_path)
        if os.path.isdir(os.path.join(parent_path, folder_name)) and '_' in folder_name
        if (version_texts := load_versions(os.path.join(parent_path, folder_name)))
    }

def generate_versions(folder_path):
    for filename in os.listdir(folder_path):
        full_path = os.path.join(folder_path, filename)

        if filename.endswith('_C.txt'):
            with open(full_path, 'r', encoding='utf-8') as f:
                yield 0, f.read()
        
        elif '_C-' in filename and filename.endswith('.txt'):
            if (match := re.search(r'_C-(\d+)\.txt', filename)):
                k = int(match.group(1))
                if k > 0 and k % 3 == 0:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        yield k, f.read()

def load_versions(folder_path):
    return dict(generate_versions(folder_path))


def text_to_words(text):
    words = re.split(r'\s+', text.strip())
    return words


def get_shingles_and_signature(words, w, lambda_val):
    if len(words) < w:
        return set()

    unique_shingles = {' '.join(words[i:i + w]) for i in range(len(words) - w + 1)}

    hashes = [
        int.from_bytes(hashlib.md5(sh.encode('utf-8')).digest(), 'big')
        for sh in unique_shingles
    ]

    sorted_hashes = sorted(hashes)
    
    if lambda_val == float('inf'):
        signature = sorted_hashes
    else:
        signature = sorted_hashes[:lambda_val]

    return set(signature)


def jaccard_similarity_calculator(set1, set2):
    union_size = len(set1.union(set2))
    
    if union_size == 0:
        return 0.0
    
    intersection_size = len(set1.intersection(set2))
    return intersection_size / union_size


def measure_timings(all_words, pairs, num_runs):
    timings = {}
    for w, lam in pairs:
        run_times = []
        for i in range(num_runs):
            start = time.time()
            for words in all_words:
                get_shingles_and_signature(words, w, lam)
            end = time.time()
            run_times.append(end-start)
            print(f'Timing Run {i+1}/{num_runs} for (w={w}, λ={lam if lam != float("inf") else "∞"}) completed.')
        
        avg_time = sum(run_times) / len(run_times)
        timings[(w, lam)] = avg_time
        print(f"  -> Average Time: {avg_time:.4f} seconds\n")
    return timings

def calculate_all_similarities(city_data, pairs):
    similarities = defaultdict(lambda: defaultdict(dict))
    for city_name, data in city_data.items():
        current_k = 0
        if current_k not in data['words']:
            continue
        
        older_ks = sorted([k for k in data['words'] if k > 0])
        version_words = data['words']
        
        for w, lam in pairs:
            current_shingles = get_shingles_and_signature(version_words[current_k], w, lam)
            sims = {
                k: jaccard_similarity_calculator(
                    current_shingles, get_shingles_and_signature(version_words[k], w, lam)
                ) for k in older_ks
            }
            similarities[(w, lam)][city_name] = sims
        print(f'Similarity calculation for {city_name} completed.')
    return similarities

def analyze_best_lambda(similarities, city_data, ws, lambdas):
    for w in ws:
        inf_sims = similarities[(w, float('inf'))]
        results = {}
        for lam in lambdas[:-1]:
            lam_sims = similarities[(w, lam)]

            diffs = [
                abs(lam_sims[city][k] - inf_sims[city][k])
                for city in city_data if city in lam_sims and city in inf_sims
                for k in lam_sims[city]
            ]
            
            avg_diff = sum(diffs) / len(diffs) if diffs else float('inf')
            results[lam] = avg_diff
            print(f"Average difference for w={w}, λ={lam}: {avg_diff:.4f}")

        if results:
            best_lambda = min(results, key=results.get)
            print(f"Best λ for w={w} (closest to ∞): {best_lambda}\n")
        else:
            print(f"Could not determine best λ for w={w}.\n")

#########################################################
################## Running Experiments ##################
#########################################################

def run_experiments(parent_path):
    city_data = load_cities(parent_path)
    if not city_data:
        print("No city data found.")
        return
    else:
        print("Imported city data from the corpus!\n")

    all_version_words = []
    for city in city_data.values():
        all_version_words.extend(city['words'].values())

    ws = [25, 50]
    lambdas = [8, 16, 32, 64, float('inf')]
    pairs = [(w, lam) for w in ws for lam in lambdas]

    timings = {}
    num_runs = 8
    timings = measure_timings(all_version_words, pairs, num_runs)
    print("\n------------------------------------------------------\n")
    similarities = calculate_all_similarities(city_data, pairs)
    print("\n------------------------------------------------------\n")
    analyze_best_lambda(similarities, city_data, ws, lambdas)
    print("------------------------------------------------------")

    states = defaultdict(list)
    for city_name in city_data:
        state = city_name.split('_')[-1]
        states[state].append(city_name)

    #########################################################
    ################# Plotting Similarities #################
    #########################################################

    for city_name in city_data:
        fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=True)

        w = 25
        ax_left = axes[0]
        for lam in lambdas:
            label = f'λ={lam}' if lam != float('inf') else 'λ=∞ (True Similarity)'
            sims = similarities[(w, lam)][city_name]
            if not sims: continue
            
            x = sorted(sims.keys())
            y = [sims[k] for k in x]

            sns.lineplot(x=x, y=y, label=label, ax=ax_left)
        

        ax_left.set_title(f'Shingle Size w = {w}')
        ax_left.set_xlabel('Version (T-k)')
        ax_left.set_ylabel('Jaccard Similarity')
        ax_left.legend()
        ax_left.grid(True, linestyle="--", alpha=0.5)


        w = 50
        ax_right = axes[1]
        for lam in lambdas:
            label = f'λ={lam}' if lam != float('inf') else 'λ=∞ (True Similarity)'
            sims = similarities[(w, lam)][city_name]
            if not sims: continue

            x = sorted(sims.keys())
            y = [sims[k] for k in x]
            sns.lineplot(x=x, y=y, label=label, ax=ax_right)

        ax_right.set_title(f'Shingle Size w = {w}')
        ax_right.set_xlabel('Version (T-k)')
        ax_right.legend()
        ax_right.grid(True, linestyle="--", alpha=0.5)


        fig.suptitle(f'Similarity Evolution for {city_name}', fontsize=20)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f'similarities_{city_name}.png')
        plt.close()


    #########################################################
    ################### Plotting Timings ####################
    #########################################################

    plot_data = []
    
    lambda_values = [8, 16, 32, 64, float('inf')]
    lambda_labels = [str(int(l)) if l != float('inf') else '∞' for l in lambda_values]

    for (w, lam), avg_time in timings.items():
        x_position = lambda_values.index(lam)
        plot_data.append({'w': f'w={w}', 'λ_pos': x_position, 'Time (seconds)': avg_time})
    
    df_timings = pd.DataFrame(plot_data)
    
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.lineplot(
        data=df_timings,
        x='λ_pos',
        y='Time (seconds)',
        hue='w',
        style='w',
        markers=True,
        dashes=False,
        ax=ax
    )

    ax.set_xticks(range(len(lambda_labels)))
    ax.set_xticklabels(lambda_labels)

    ax.set_title('Timings for Shingle Calculation (Entire Corpus)', fontsize=14)
    ax.set_xlabel('λ Value')
    ax.set_ylabel('Time (seconds)')
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(title=None)

    plt.tight_layout()
    plt.savefig('Timings.png')

    
if __name__ == "__main__":
    parent_path = 'DataDump'
    run_experiments(parent_path)
