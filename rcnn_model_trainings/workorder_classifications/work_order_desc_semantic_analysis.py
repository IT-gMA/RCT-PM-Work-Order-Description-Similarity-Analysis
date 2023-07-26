import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor

# pytorch library
import torch  # the main pytorch library
import torch.nn.functional as f  # the sub-library containing different functions for manipulating with tensors

# huggingface's transformers library
from transformers import BertModel, BertTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "mps")

bert_version = 'bert-large-uncased'
tokenizer = BertTokenizer.from_pretrained(bert_version)

model = BertModel.from_pretrained(bert_version)
model = model.eval()
model = model.to(device)

texts = [
    'Grapefruit Pink (IMPORTED)',
    'Grapefruit Pink KG (IMPORTED)',
    'Grapes Red 10kg Carton ',
    'Grape White 10kg CARTON ',
    'Kiwi Fruit EA (IMPORTED)'
]

encodings = tokenizer(
    texts,  # the texts to be tokenized
    padding=True,  # pad the texts to the maximum length (so that all outputs have the same length)
    return_tensors='pt'  # return the tensors (not lists)
)

encodings = encodings.to(device)

with torch.no_grad():
    # get the model embeddings
    embeds = model(**encodings)


def visualize(distances, figsize=(10, 5), titles=None):
    # get the number of columns
    ncols = len(distances)
    # create the subplot placeholders
    fig, ax = plt.subplots(ncols=ncols, figsize=figsize)

    for i in range(ncols):
        # get the axis in which we will draw the matrix
        axes = ax[i] if ncols > 1 else ax

        # get the i-th distance
        distance = distances[i]

        # create the heatmap
        axes.imshow(distance)

        # show the ticks
        axes.set_xticks(np.arange(distance.shape[0]))
        axes.set_yticks(np.arange(distance.shape[1]))

        # set the tick labels
        axes.set_xticklabels(np.arange(distance.shape[0]))
        axes.set_yticklabels(np.arange(distance.shape[1]))

        # set the values in the heatmap
        for j in range(distance.shape[0]):
            for k in range(distance.shape[1]):
                text = axes.text(k, j, str(round(distance[j, k], 3)),
                                 ha="center", va="center", color="w")

        # set the title of the subplot
        title = titles[i] if titles and len(titles) > i else "Text Distance"
        axes.set_title(title, fontsize="x-large")

    fig.tight_layout()
    plt.show()


def main():
    CLSs = embeds[0][:, 0, :]
    # normalize the CLS token embeddings
    normalized = f.normalize(CLSs, p=2, dim=1)
    # calculate the cosine similarity
    cls_dist = normalized.matmul(normalized.T)
    cls_dist = cls_dist.new_ones(cls_dist.shape) - cls_dist
    cls_dist = cls_dist.cpu().numpy()

    visualize([cls_dist], titles=["CLS"])


if __name__ == '__main__':
    main()
