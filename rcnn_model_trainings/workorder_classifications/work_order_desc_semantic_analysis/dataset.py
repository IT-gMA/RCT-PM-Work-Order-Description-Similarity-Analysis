from util_fucntions import util_functions
from configs import TRAIN_RATIO, VALIDATION_RATIO, TEST_RATIO, DATA_FILE_PATH, RANDOM_SEED, TRAIN_BATCH_SIZE, VAL_BATCH_SIZE
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset


class WorkOrderDescriptionSemanticDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        # Extract the label and any other necessary data from the dictionary
        y_true = sample['similarity']
        # ... extract other data from the dictionary if needed ...
        return y_true, sample  # Return the label and the dictionary sample


def _even_out_similarities(excel_data: list) -> list:
    exacts = []
    similars = []
    neutrals = []

    print(f'Looking at {len(excel_data)} similarity values:')
    for data in tqdm(excel_data):
        if data['similarity'] >= .9:
            exacts.append(data)
        elif data['similarity'] <= .1:
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
    train_loader = DataLoader(
        dataset=WorkOrderDescriptionSemanticDataset(train_set),
        batch_size=TRAIN_BATCH_SIZE,
        num_workers=1
    )

    validation_loader = DataLoader(
        dataset=WorkOrderDescriptionSemanticDataset(validation_set),
        batch_size=VAL_BATCH_SIZE,
        num_workers=1
    )

    test_loader = DataLoader(
        dataset=WorkOrderDescriptionSemanticDataset(test_set),
        batch_size=VAL_BATCH_SIZE,
        num_workers=1
    )
    return train_loader, validation_loader, test_loader


def main():
    train_set, val_test, test_set = get_splitted_dataset()
    train_loader, validation_loader, test_loader = get_data_loaders(train_set, val_test, test_set)


if __name__ == '__main__':
    main()

