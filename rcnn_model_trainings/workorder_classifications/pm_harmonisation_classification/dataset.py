from util_fucntions import util_functions
from configs import *
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import torch
from itertools import groupby
from operator import itemgetter


def format_label_indexes(label_indexes: list) -> list:
    return [{'idx': idx, 'label': label.item()} for idx, label in label_indexes]


class TextClassificationDataset(Dataset):
    def __init__(self, data, tokenizer, max_length, classes: list):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

        json_class_data = get_static_classes_data()
        self.classes = json_class_data[STATIC_CLASS_LABEL_KEY_NAME]
        self.num_classes = len(self.classes)
        self.label_to_index = json_class_data[STATIC_ENUMERATED_CLASS_DATA_NAME]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        input_text = item[INPUT_KEY_NAME]
        label_text = item[LABEL_KEY_NAME]

        encoding = self.tokenizer.encode_plus(
            input_text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        # print(f'maximo desc: {input_text}\tlabel text: {label_text}')

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'harmonised_desc': label_text
        }

    def get_label_index(self, labels: list):
        return torch.tensor([self.label_to_index[label] for label in labels]).to(DEVICE)

    def get_label_strings(self, index_tensors) -> list:
        if DEVICE != 'cuda':
            index_tensors = index_tensors.cpu()
        return [self.label_to_index[idx.item()] for idx in index_tensors]

    def get_num_classes(self) -> int:
        return self.num_classes

    def format_label_to_index(self) -> list:
        return [{'idx': idx, 'label': label} for label, idx in self.label_to_index.items()]


def _even_out_labels(excel_data: list) -> list:
    util_functions.random_seed_shuffle(seed=int(RANDOM_SEED / 1.5), og_list=excel_data)
    excel_data.sort(key=itemgetter('harmonised_desc'))
    return [{
        'grouped_label': key,
        'grouped_data': list(value),
    } for key, value in groupby(excel_data, lambda x: x['harmonised_desc'])]


def get_splitted_dataset() -> tuple:
    train_set = []
    validation_set = []
    test_set = []

    grouped_by_label_datas = _even_out_labels(util_functions.read_excel_file(path=DATA_FILE_PATH, format_key=True))

    for grouped_by_label_data in tqdm(grouped_by_label_datas):
        grouped_data = grouped_by_label_data['grouped_data']
        util_functions.random_seed_shuffle(seed=RANDOM_SEED, og_list=grouped_data)
        num_trains = int(len(grouped_data) * TRAIN_RATIO)
        num_vals = int(len(grouped_data) * VALIDATION_RATIO)

        train_set.append(grouped_data[0:num_trains])
        validation_set.append(grouped_data[num_trains:num_trains + num_vals])
        test_set.append(grouped_data[num_trains + num_vals:len(grouped_data)])

    train_set = util_functions.flatten_list(train_set)
    validation_set = util_functions.flatten_list(validation_set)
    test_set = util_functions.flatten_list(test_set)

    for _set in [train_set, validation_set, test_set]:
        util_functions.random_seed_shuffle(seed=int(RANDOM_SEED * 1.5), og_list=_set)
    return train_set, validation_set, test_set, sorted([data['grouped_label'] for data in grouped_by_label_datas])


def get_data_loaders(train_set: list, validation_set: list, test_set: list, classes: list) -> tuple:
    samples_used_for_training = [data[SAMPLE_IDX_CODE_NAME] for data in train_set]
    util_functions.write_to_json_file(samples_used_for_training, SAVED_TRAINED_SAMPLE_IDX_LOCATION)

    train_loader = DataLoader(
        dataset=TextClassificationDataset(train_set, MODEL_TOKENIZER, MAX_LENGTH_TOKEN, classes),
        batch_size=TRAIN_BATCH_SIZE,
        num_workers=NUM_WORKERS
    )

    validation_loader = DataLoader(
        dataset=TextClassificationDataset(validation_set, MODEL_TOKENIZER, MAX_LENGTH_TOKEN, classes),
        batch_size=VAL_BATCH_SIZE,
        num_workers=NUM_WORKERS
    )

    test_loader = DataLoader(
        dataset=TextClassificationDataset(test_set, MODEL_TOKENIZER, MAX_LENGTH_TOKEN, classes),
        batch_size=VAL_BATCH_SIZE,
        num_workers=NUM_WORKERS
    )
    return train_loader, validation_loader, test_loader


def _write_enumerated_classes_to_json(classes: list):
    classes.sort()
    classes = [{STATIC_CLASS_IDX_KEY_NAME: index,
                STATIC_CLASS_LABEL_KEY_NAME: label} for index, label in enumerate(classes)]
    for data in classes:
        print(data)
    util_functions.write_to_json_file(data_list=classes, file_path=STATIC_CLASS_LABEL_FILE_LOCATION)


def _read_static_json_classes():
    class_json_data = util_functions.read_from_json_file(STATIC_CLASS_LABEL_FILE_LOCATION)
    for data in class_json_data:
        print(data)


def get_enumerated_class_idx() -> list:
    return sorted([data[STATIC_CLASS_IDX_KEY_NAME] for data in
                   util_functions.read_from_json_file(STATIC_CLASS_LABEL_FILE_LOCATION)])


def get_static_class_labels() -> list:
    return [data[STATIC_CLASS_LABEL_KEY_NAME] for data in
            sorted(util_functions.read_from_json_file(STATIC_CLASS_LABEL_FILE_LOCATION),
                   key=itemgetter(STATIC_CLASS_IDX_KEY_NAME))]


def get_static_classes_data() -> dict:
    json_class_data = sorted(util_functions.read_from_json_file(STATIC_CLASS_LABEL_FILE_LOCATION),
                             key=itemgetter(STATIC_CLASS_IDX_KEY_NAME))
    return {STATIC_ENUMERATED_CLASS_DATA_NAME: util_functions.list_of_dicts_to_single_dict(list_of_dicts=json_class_data, key_name=STATIC_CLASS_LABEL_KEY_NAME, value_name=STATIC_CLASS_IDX_KEY_NAME),
            STATIC_CLASS_IDX_KEY_NAME: [data[STATIC_CLASS_IDX_KEY_NAME] for data in json_class_data],
            STATIC_CLASS_LABEL_KEY_NAME: [data[STATIC_CLASS_LABEL_KEY_NAME] for data in json_class_data], }


def main():
    train_set, val_test, test_set, classes = get_splitted_dataset()
    train_loader, validation_loader, test_loader = get_data_loaders(train_set, val_test, test_set, classes)
    # samples_not_used_for_training = util_functions.read_from_json_file(SAVED_UNTRAINED_SAMPLE_IDX_LOCATION)
    # samples_used_for_training = util_functions.read_from_json_file(SAVED_TRAINED_SAMPLE_IDX_LOCATION)
    # print(f'there are {len(samples_used_for_training)} used for training:\n{samples_used_for_training}')


if __name__ == '__main__':
    main()
