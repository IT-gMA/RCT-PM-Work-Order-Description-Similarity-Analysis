from util_fucntions import util_functions
from configs import *
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import torch
from itertools import groupby
from operator import itemgetter


class TextClassificationDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

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

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'harmonised_desc': label_text
        }


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


def get_data_loaders(train_set: list, validation_set: list, test_set: list) -> tuple:
    samples_used_for_training = [data[SAMPLE_IDX_CODE_NAME] for data in train_set]
    util_functions.write_to_json_file(samples_used_for_training, SAVED_TRAINED_SAMPLE_IDX_LOCATION)

    train_loader = DataLoader(
        dataset=TextClassificationDataset(train_set, MODEL_TOKENIZER, MAX_LENGTH_TOKEN),
        batch_size=TRAIN_BATCH_SIZE,
        num_workers=NUM_WORKERS
    )

    validation_loader = DataLoader(
        dataset=TextClassificationDataset(validation_set, MODEL_TOKENIZER, MAX_LENGTH_TOKEN),
        batch_size=VAL_BATCH_SIZE,
        num_workers=NUM_WORKERS
    )

    test_loader = DataLoader(
        dataset=TextClassificationDataset(test_set, MODEL_TOKENIZER, MAX_LENGTH_TOKEN),
        batch_size=VAL_BATCH_SIZE,
        num_workers=NUM_WORKERS
    )
    return train_loader, validation_loader, test_loader


def main():
    train_set, val_test, test_set, classes = get_splitted_dataset()
    train_loader, validation_loader, test_loader = get_data_loaders(train_set, val_test, test_set)
    # samples_not_used_for_training = util_functions.read_from_json_file(SAVED_UNTRAINED_SAMPLE_IDX_LOCATION)
    # samples_used_for_training = util_functions.read_from_json_file(SAVED_TRAINED_SAMPLE_IDX_LOCATION)
    # print(f'there are {len(samples_used_for_training)} used for training:\n{samples_used_for_training}')


if __name__ == '__main__':
    main()
