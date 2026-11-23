"""Rebuild current study figures from checked evidence, without changing results or README."""
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from check_results import ROOT, STUDY, MODELS, SEEDS, check, read_csv

LABELS = ['Transformer U-Net', 'Attention + Transformer U-Net']
COLORS = ['#4267A8', '#008577']


def main():
    records = check()
    output = ROOT / 'docs/images'
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.facecolor': 'white', 'axes.titleweight': 'bold'})
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5))
    for ax, key, title in zip(axes, ['dice_mean', 'iou_mean'], ['Test Dice', 'Test IoU']):
        for seed in SEEDS:
            ax.plot([0, 1], [100*records[m, seed][key] for m in MODELS], '.--',
                    alpha=.65, label=f'Seed {seed}')
        for i, model in enumerate(MODELS):
            values = [100*records[model, s][key] for s in SEEDS]
            ax.errorbar(i, statistics.mean(values), yerr=statistics.stdev(values), fmt='o',
                        capsize=6, color=COLORS[i], markersize=8, zorder=5)
        ax.set(xticks=[0, 1], xticklabels=['Transformer', 'Attention +\nTransformer'],
               ylabel='Score (%)', title=title, xlim=(-.35, 1.35))
        ax.grid(axis='y', alpha=.18)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle('Paired comparison on the same 150 test images', fontsize=14, weight='bold')
    fig.text(.5, .02, 'Large points: mean; bars: sample SD across three seeds (not confidence intervals).', ha='center', fontsize=9)
    fig.tight_layout(rect=[0, .06, 1, .95])
    fig.savefig(output / 'seed_study_test_comparison.png', dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
    for ax, model, label in zip(axes, MODELS, LABELS):
        for seed in SEEDS:
            rows = read_csv(STUDY / f'{model}-seed{seed}.history.csv')
            epochs = [int(r['epoch']) for r in rows]
            values = [100*float(r['val_dice']) for r in rows]
            line, = ax.plot(epochs, values, linewidth=1.5, label=f'Seed {seed}')
            best = max(range(len(values)), key=values.__getitem__)
            ax.scatter(epochs[best], values[best], color=line.get_color(), marker='*', s=85, zorder=4)
        ax.set(title=label, xlabel='Epoch', xlim=(1, 100), ylim=(0, 100))
        ax.grid(alpha=.18)
        ax.legend(fontsize=8, frameon=False, loc='lower right')
    axes[0].set_ylabel('Validation Dice (%)')
    fig.suptitle('Validation trajectories across all six runs', fontsize=14, weight='bold')
    fig.text(.5, .02, 'Stars mark each selected best epoch. Curves stop at the last recorded epoch.', ha='center', fontsize=9)
    fig.tight_layout(rect=[0, .06, 1, .95])
    fig.savefig(output / 'seed_study_learning_curves.png', dpi=170)
    plt.close(fig)
    print('Updated the two seed-study figures. Measurements and README were not modified.')


if __name__ == '__main__':
    main()
