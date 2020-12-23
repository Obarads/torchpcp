import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

try:
    from tsnecuda import TSNE
except:
    from sklearn.manifold import TSNE

def write_tsne(label_dict, embedding, extension="png", tsne_model=None):
    """
    base:https://medium.com/analytics-vidhya/super-fast-tsne-cuda-on-kaggle-b66dcdc4a5a4
    """
    if tsne_model is None:
        tsne_model = TSNE()

    x_embedding = tsne_model.fit_transform(embedding)

    for key in label_dict:
        label = label_dict[key]
        embedding_and_label = pd.concat([pd.DataFrame(x_embedding), pd.DataFrame(data=label,columns=["label"])], axis=1)
        sns.FacetGrid(embedding_and_label, hue="label", height=6).map(plt.scatter, 0, 1).add_legend()
        plt.savefig("{}.{}".format(key,extension))
        plt.clf()
    plt.close('all')
