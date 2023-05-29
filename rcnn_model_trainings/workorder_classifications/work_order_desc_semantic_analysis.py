import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor

# pytorch library
import torch  # the main pytorch library
import torch.nn.functional as f  # the sub-library containing different functions for manipulating with tensors

# huggingface's transformers library
from transformers import BertModel, BertTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "mps")

bert_version = 'bert-large-cased'
tokenizer = BertTokenizer.from_pretrained(bert_version)

model = BertModel.from_pretrained(bert_version)
model = model.eval()
model = model.to(device)

texts = [
    'Unit 4, Cameras in perimeter zones 1,3,4,5,6 & PTZ 38 are not working',
    '3 Monthly - CCTV System - Head End - Server/Workstation/Software/Licence - Inspect, Update & Report',
    '3 Monthly - CCTV System  - Head End - Monitor -Inspect, Clean & Report',
    '3M - CCTV System  - Camera - Internal  -Inspect, Clean & Report',
    'Annually-Thermostatic Mixing Valve - Inspection, Service & Report',
    '3 Monthly - CCTV System  - Head End - Recorder -Inspect, Clean & Report',
    '3M - CCTV System  - Camera - PTZ - Inspect, Clean & Report',
    '3M - CCTV System  - Power Supply  - Inspect, Clean & Report',
    'Monthly-Irrigation - Inspection, Service and Report',
    '3M - CCTV System  - Encoder / Decoder  - Inspect, Clean & Report',
    '3M - Security System: CCTV Camera / External - Inspection, Clean & Report',
    '3M - Security System: CCTV Video Modem / Media Converter - Inspection, Clean & Report',
    '3M - Security System: CCTV Video Distribution / Multiplexer / Quad - Inspection, Clean & Report',
    'Quarterly-Steam Boilers - Inspection and Report',
]

encodings = tokenizer(
    texts,  # the texts to be tokenized
    padding=True,  # pad the texts to the maximum length (so that all outputs have the same length)
    return_tensors='pt'  # return the tensors (not lists)
)

encodings = encodings.to(device)

for tokens in encodings['input_ids']:
    print(tokenizer.convert_ids_to_tokens(tokens))

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
    cls_dist = cls_dist.cpu()
    #cls_dist

    visualize([cls_dist], titles=["CLS"])


if __name__ == '__main__':
    main()
