from util_fucntions import util_functions
from configs import *
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import torch
from itertools import groupby
from operator import itemgetter

GROUPED_DATA_KEY_NAME = 'grouped_data'


class WorkOrderDescriptionSemanticDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Extract the text and similarity score from the dictionary
        text1 = str(item[TEXT1_KEY_NAME])
        text2 = str(item[TEXT2_KEY_NAME])
        similarity = item[ACTUAL_VALUE_KEY_NAME]
        if type(similarity) != float:
            similarity = float(similarity)

        # Tokenize the text pair using the BERT tokenizer
        inputs = self.tokenizer.encode_plus(
            text1,
            text2,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        # Convert the similarity score to a torch.Tensor
        similarity = torch.tensor(similarity, dtype=torch.float32)

        # Return the tokenized inputs and the similarity score
        _item = {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'similarity': similarity.squeeze(),
        }
        if IS_BERT:
            token_type_ids = inputs.get('token_type_ids', None)  # Obtain the token_type_ids
            if token_type_ids is not None:
                token_type_ids = token_type_ids.squeeze()
            _item['token_type_ids'] = token_type_ids

        return _item


def _even_out_similarities(excel_data: list) -> list:
    exacts = []
    similars = []
    neutrals = []

    print(f'Looking at {len(excel_data)} similarity values:')
    for data in tqdm(excel_data):
        if data['similarity'] >= .75:
            exacts.append(data)
        elif data['similarity'] <= .3:
            neutrals.append(data)
        else:
            similars.append(data)
    return [exacts, similars, neutrals]


def _even_out_pm_desc(excel_data: list) -> list:
    util_functions.random_seed_shuffle(seed=int(RANDOM_SEED / 1.5), og_list=excel_data)
    return [{
        TEXT1_KEY_NAME: key,
        GROUPED_DATA_KEY_NAME: list(value)
    } for key, value in groupby(sorted(excel_data, key=itemgetter(TEXT2_KEY_NAME)), lambda x: x[TEXT2_KEY_NAME])]


def get_splitted_dataset() -> tuple:
    train_set = []
    validation_set = []
    test_set = []

    excel_data = util_functions.read_excel_file(path=DATA_FILE_PATH, format_key=True)
    util_functions.random_seed_shuffle(seed=RANDOM_SEED, og_list=excel_data)
    distributed_desc_list = _even_out_similarities(excel_data)

    for distributed_desc in tqdm(distributed_desc_list):
        # distributed_desc = distributed_desc[GROUPED_DATA_KEY_NAME]
        util_functions.random_seed_shuffle(seed=int(RANDOM_SEED * 1.5), og_list=distributed_desc)

        num_trains = int(len(distributed_desc) * TRAIN_RATIO)
        num_vals = int(len(distributed_desc) * VALIDATION_RATIO)

        train_set.append(distributed_desc[0:num_trains])
        validation_set.append(distributed_desc[num_trains:num_trains + num_vals])
        test_set.append(distributed_desc[num_trains + num_vals:len(distributed_desc)])

    train_set = util_functions.flatten_list(train_set)
    validation_set = util_functions.flatten_list(validation_set)
    test_set = util_functions.flatten_list(test_set)
    for _set in [train_set, validation_set, test_set]:
        util_functions.random_seed_shuffle(seed=int(RANDOM_SEED * 1.5), og_list=_set)

    return train_set, validation_set, test_set


def get_data_loaders(train_set: list, validation_set: list, test_set: list) -> tuple:
    # samples_not_used_for_training = sorted(list(set([sample[SAMPLE_IDX_CODE_NAME].split(':')[0] for sample in util_functions.flatten_list([validation_set, test_set])])))
    samples_used_for_training = util_functions.get_unique_list(
        old_list=[sample[SAMPLE_IDX_CODE_NAME].split(':')[0] for sample in train_set], sort_code=1)
    util_functions.write_to_json_file(samples_used_for_training, SAVED_TRAINED_SAMPLE_IDX_LOCATION)

    train_loader = DataLoader(
        dataset=WorkOrderDescriptionSemanticDataset(train_set, MODEL_TOKENIZER, MAX_LENGTH_TOKEN),
        batch_size=TRAIN_BATCH_SIZE,
        num_workers=NUM_WORKERS
    )

    validation_loader = DataLoader(
        dataset=WorkOrderDescriptionSemanticDataset(validation_set, MODEL_TOKENIZER, MAX_LENGTH_TOKEN),
        batch_size=VAL_BATCH_SIZE,
        num_workers=NUM_WORKERS
    )

    test_loader = DataLoader(
        dataset=WorkOrderDescriptionSemanticDataset(test_set, MODEL_TOKENIZER, MAX_LENGTH_TOKEN),
        batch_size=VAL_BATCH_SIZE,
        num_workers=NUM_WORKERS
    )
    return train_loader, validation_loader, test_loader


def main():
    train_set, val_test, test_set = get_splitted_dataset()
    print(len(train_set))
    print(len(test_set))
    train_loader, validation_loader, test_loader = get_data_loaders(train_set, val_test, test_set)
    # samples_not_used_for_training = util_functions.read_from_json_file(SAVED_UNTRAINED_SAMPLE_IDX_LOCATION)
    # samples_used_for_training = util_functions.read_from_json_file(SAVED_TRAINED_SAMPLE_IDX_LOCATION)
    # print(f'there are {len(samples_used_for_training)} used for training:\n{samples_used_for_training}')


if __name__ == '__main__':
    main()
