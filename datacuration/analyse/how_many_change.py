import matplotlib.pyplot as plt
import fire
from careerpathway.scoring.load_testset import load_diversity

def draw(
    figsize: tuple = (6, 1.5),
    fontsize: int = 15,
    bincolor: str = '#8E04FB',
):
    testset, _ = load_diversity(initial_node_idx=0)
    lens = [len(item['nodes'])+1 for item in testset]
    print(max(lens))
    plt.figure(figsize=figsize)
    plt.hist(lens, bins=max(lens) // 2, color=bincolor)
    plt.text(0.95, 0.9, f'Mean: {sum(lens)/len(lens):.1f}\nMax: {max(lens)}', transform=plt.gca().transAxes, fontsize=fontsize, verticalalignment='top', horizontalalignment='right')
    
    plt.xlim(1, 40)
    plt.xticklabels = range(1, max(lens))
    plt.xlabel('Number of items in profile', fontsize=fontsize)
    # remove top and right border
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.savefig('results/plots/len_items.png', bbox_inches='tight')
    plt.close()
    return


if __name__ == '__main__':
    fire.Fire(draw)

