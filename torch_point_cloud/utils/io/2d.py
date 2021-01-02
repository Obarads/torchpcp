import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def write_embeddings(label_dict, embedding, extension="png", method=""):
    """
    base:https://medium.com/analytics-vidhya/super-fast-tsne-cuda-on-kaggle-b66dcdc4a5a4
    """

    if method == "tsne":
        try:
            from tsnecuda import TSNE
        except:
            from sklearn.manifold import TSNE
        model = TSNE()
    elif method == "umap":
        # https://umap-learn.readthedocs.io/en/latest/
        from umap import UMAP
        model = UMAP()
    else:
        raise NotImplementedError("Unknown method: {}".format(method))

    x_embedding = model.fit_transform(embedding)

    for key in label_dict:
        label = label_dict[key]
        embedding_and_label = pd.concat([pd.DataFrame(x_embedding), pd.DataFrame(data=label,columns=["label"])], axis=1)
        sns.FacetGrid(embedding_and_label, hue="label", height=6).map(plt.scatter, 0, 1).add_legend()
        plt.savefig("{}.{}".format(key,extension))
        plt.clf()
    plt.close('all')

