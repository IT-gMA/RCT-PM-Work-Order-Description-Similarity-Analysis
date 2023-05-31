from util_fucntions import util_functions
from configs import TRAIN_RATIO, VALIDATION_RATIO, TEST_RATIO, DATA_FILE_PATH, RANDOM_SEED, TRAIN_BATCH_SIZE, VAL_BATCH_SIZE, ACTUAL_VALUE_KEY_NAME, BERT_TOKENIZER, MAX_LENGTH_TOKEN, TEXT1_KEY_NAME, TEXT2_KEY_NAME, RUNNING_LOG_LOCATION, SAMPLE_IDX_CODE_NAME, SAVED_UNTRAINED_SAMPLE_IDX_LOCATION, SAVED_TRAINED_SAMPLE_IDX_LOCATION
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import torch


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
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        # Obtain the token_type_ids
        token_type_ids = inputs.get('token_type_ids', None)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.squeeze()

        # Convert the similarity score to a torch.Tensor
        similarity = torch.tensor(similarity, dtype=torch.float32)

        # Return the tokenized inputs and the similarity score
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'token_type_ids': token_type_ids,
            'similarity': similarity,
        }


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


def get_splitted_dataset() -> tuple:
    train_set = []
    validation_set = []
    test_set = []

    excel_data = util_functions.read_excel_file(path=DATA_FILE_PATH, format_key=True)
    util_functions.random_seed_shuffle(seed=RANDOM_SEED, og_list=excel_data)
    distributed_desc_list = _even_out_similarities(excel_data)

    num_trains = int(len(excel_data) * TRAIN_RATIO)
    num_vals = int(len(excel_data) * VALIDATION_RATIO)
    num_tests = len(excel_data) - num_vals - num_trains

    print(f'Splitting train: {num_trains}\tvalidation: {num_vals}\ttest: {num_tests}')
    for distributed_desc in tqdm(distributed_desc_list):
        train_set.append(distributed_desc[0:num_trains])
        validation_set.append(distributed_desc[num_trains:num_trains + num_vals])
        test_set.append(distributed_desc[num_trains + num_vals:len(excel_data)])
        '''[train_set.append(data) for data in distributed_desc[0:num_trains]]
        [validation_set.append(data) for data in distributed_desc[num_trains:num_trains + num_vals]]
        [test_set.append(data) for data in distributed_desc[num_trains + num_vals:len(excel_data)]]'''

    return util_functions.flatten_list(train_set), util_functions.flatten_list(validation_set), util_functions.flatten_list(test_set)


def get_data_loaders(train_set: list, validation_set: list, test_set: list) -> tuple:
    #samples_not_used_for_training = sorted(list(set([sample[SAMPLE_IDX_CODE_NAME].split(':')[0] for sample in util_functions.flatten_list([validation_set, test_set])])))
    samples_used_for_training = util_functions.get_unique_list(old_list=[sample[SAMPLE_IDX_CODE_NAME].split(':')[0] for sample in train_set], sort_code=1)
    #util_functions.write_to_json_file(samples_used_for_training, SAVED_TRAINED_SAMPLE_IDX_LOCATION)

    train_loader = DataLoader(
        dataset=WorkOrderDescriptionSemanticDataset(train_set, BERT_TOKENIZER, MAX_LENGTH_TOKEN),
        batch_size=TRAIN_BATCH_SIZE,
        num_workers=1
    )

    validation_loader = DataLoader(
        dataset=WorkOrderDescriptionSemanticDataset(validation_set, BERT_TOKENIZER, MAX_LENGTH_TOKEN),
        batch_size=VAL_BATCH_SIZE,
        num_workers=1
    )

    test_loader = DataLoader(
        dataset=WorkOrderDescriptionSemanticDataset(test_set, BERT_TOKENIZER, MAX_LENGTH_TOKEN),
        batch_size=VAL_BATCH_SIZE,
        num_workers=1
    )
    return train_loader, validation_loader, test_loader


def main():
    train_set, val_test, test_set = get_splitted_dataset()
    train_loader, validation_loader, test_loader = get_data_loaders(train_set, val_test, test_set)
    print(len(validation_loader))
    print(len(test_loader))
    #samples_not_used_for_training = util_functions.read_from_json_file(SAVED_UNTRAINED_SAMPLE_IDX_LOCATION)
    #samples_used_for_training = util_functions.read_from_json_file(SAVED_TRAINED_SAMPLE_IDX_LOCATION)
    #print(f'there are {len(samples_used_for_training)} used for training:\n{samples_used_for_training}')


if __name__ == '__main__':
    main()

