from util_fucntions import util_functions
from tqdm import tqdm
import itertools
from itertools import groupby
from operator import itemgetter

EXCEL_IMPORT_FILE_PATHS = ['../xlsx_resources/downloaded_mappings/DOEN_uploaded_data.xlsx',
                           '../xlsx_resources/downloaded_mappings/DOES_uploaded_data.xlsx']
OG_JSON_PERMUTATION_PATH = '../rcnn_model_trainings/workorder_classifications/pm_harmonisation_classification/saved_permutaions_uid.json'
SAVED_JSON_PERMUTATION_PATH = '../rcnn_model_trainings/workorder_classifications/pm_harmonisation_classification/saved_permutaions_uid_2.json'
APPEND_NEW_DATA = True


def get_next_largest_idx(json_permutations: list) -> int:
    if len(json_permutations) < 1:
        return 0
    return sorted(json_permutations, key=itemgetter('class_idx'))[-1]['class_idx']


def retrieve_format_import_excel_data(prev_class_uids: list):
    grouped_data = []
    excel_json_data = [{'class_uid': data['dataverse_pm_uid'],
                        'class_label': data['chosen_pm_name'],
                        'main_subj': data['pm_description']}
                       for data in sorted(util_functions.flatten_list(
            [util_functions.read_excel_file(file_path, True) for file_path in EXCEL_IMPORT_FILE_PATHS]),
            key=itemgetter('dataverse_pm_uid'))
                       ]
    if len(prev_class_uids) > 0:
        print(f'filter out new classes:\t before there are {len(excel_json_data)} samples')
        excel_json_data = list(filter(lambda d: d['class_uid'] in prev_class_uids, excel_json_data))

    print(f'grouping {len(excel_json_data)} rows of excel data:\n')
    for key, value in tqdm(
            groupby(excel_json_data, lambda x: x['class_uid'])):
        value = list(value)
        grouped_data.append({
            'class_uid': str(key),
            'class_label': value[0]['class_label'],
            'main_subjs': util_functions.remove_duplicate_in_list(og_list=[v['main_subj'] for v in value],
                                                                  sort_list=True),
        })

    return sorted(grouped_data, key=itemgetter('class_label'))


def join_excel_json(grp_excel_json: list, og_json_permutaions: list) -> list:
    for json_permutaion in og_json_permutaions:
        print(f"Assessing {json_permutaion['class_label']}: {json_permutaion['class_uid']}")
        for excel_json in grp_excel_json:
            if util_functions.are_strings_the_same(excel_json['class_uid'], json_permutaion['class_uid']):
                print(excel_json)

        print(list(filter(lambda d: d['class_uid'] == json_permutaion['class_uid'], grp_excel_json)))
        excel_json = list(filter(lambda d: d['class_uid'] == json_permutaion['class_uid'], grp_excel_json))
        excel_json_main_subjs = excel_json[0]['main_subjs'] if len(excel_json) > 0 else []

        json_permutaion['main_subjs'] = util_functions.remove_duplicate_in_list(
            sorted(list(set(util_functions.flatten_list(([json_permutaion['main_subjs'],
                                                          excel_json_main_subjs]))))))
    return og_json_permutaions


def main():
    og_json_permutaions = [] if OG_JSON_PERMUTATION_PATH is None else util_functions.read_from_json_file(
        OG_JSON_PERMUTATION_PATH)
    grp_excel_json = retrieve_format_import_excel_data(
        prev_class_uids=[data['class_uid'] for data in og_json_permutaions])
    print(f'There are {len(grp_excel_json)} classes:')
    new_json_permutations = join_excel_json(grp_excel_json=grp_excel_json, og_json_permutaions=og_json_permutaions)
    util_functions.write_to_json_file(data_list=new_json_permutations, file_path=SAVED_JSON_PERMUTATION_PATH)


if __name__ == '__main__':
    main()
