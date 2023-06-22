import torch
from configs import *
from model import TextClassification
from dataset import get_splitted_dataset, get_data_loaders, get_static_classes_data
from util_fucntions import util_functions
from metrics import *
import torch.nn.functional as F

MODEL_PATH = 'saved_models/bert_based_cased_models_iteration_0/final_model.pt'
INPUT_TEXT = "EWIS"


def get_label_from_index(label_idx: int, class_dicts: list) -> str:
    corresponding_label_dict = [class_dict for class_dict in class_dicts if
                                class_dict[STATIC_CLASS_IDX_KEY_NAME] == label_idx]
    return 'undefined' if len(corresponding_label_dict) < 0 else corresponding_label_dict[0][
        STATIC_CLASS_LABEL_KEY_NAME]


def main():
    json_class_data = get_static_classes_data()
    model = TextClassification(num_classes=len(json_class_data[STATIC_CLASS_LABEL_KEY_NAME]))
    model.load_state_dict(torch.load(MODEL_PATH)['model_state_dict'])
    model.eval()
    model.to(DEVICE)

    encodings = MODEL_TOKENIZER(
        INPUT_TEXT,
        padding='max_length', max_length=MAX_LENGTH_TOKEN,
        truncation=True, return_tensors='pt'
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(encodings['input_ids'], encodings['attention_mask'])

    # Get the predicted label probabilities
    probs = F.softmax(outputs, dim=1)
    probs = probs.squeeze().tolist()

    # Get the predicted label indexes and convert them to labels
    _, predicted_labels = torch.max(outputs, dim=1)
    predicted_labels = predicted_labels.squeeze().tolist()
    print(f"predicted label: {[class_dict for class_dict in json_class_data['class_json_data'] if class_dict['class_idx'] == predicted_labels][0]['class_label']}")
    print(predicted_labels)
    predicted_labels = [get_label_from_index(label_idx=label, class_dicts=json_class_data['class_json_data']) for label
                        in predicted_labels]

    # Print the predicted labels and their confidence scores
    for label, prob in zip(predicted_labels, probs):
        print(f'Predicted: {label}\tConfidence Score: {round(prob * 100, 2)}%')


if __name__ == '__main__':
    main()
