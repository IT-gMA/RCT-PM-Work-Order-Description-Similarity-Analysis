import copy
import math
import random
from transformers import GPT2Tokenizer

from util_fucntions import util_functions
from tqdm import tqdm
import itertools
from itertools import groupby
from operator import itemgetter
import re
import requests
import nlpaug.augmenter.word as na_word
import nlpaug.augmenter.char as na_char
import nlpaug.augmenter.sentence as na_sen
import nlpaug.augmenter.word.context_word_embs as nawcwe
import nlpaug.augmenter.word.word_embs as nawwe
import nlpaug.augmenter.word.spelling as naws

MAX_TOKEN_LENGTH = 512

JSON_FILE_PATH = '../rcnn_model_trainings/workorder_classifications/pm_harmonisation_classification/saved_permutaions_uid.json'
OG_EXCEL_PATH = '../xlsx_resources/for_trainings/maximo_pm_to_gap_pm_desc_map.xlsx'
SAVED_DIR = '../xlsx_resources/for_trainings/harmonised_pm_desc_uid3'
SAVED_FILE_PATH = f'{SAVED_DIR}/harmonised_pm_desc_uid_dataset'
SAVED_SMALL_SAMPLE_FILE_PATH = '/Users/naga/Downloads/small_sample_maximo_pm_to_gap_pm_desc_map.xlsx'
WORKBOOK_HEADERS = ['idx', 'curr_desc', 'class_uid', 'harmonised_desc']
APPEND_TO_EXCEL = True
LIST_OF_WORDS = ["Testing", 'Inspection', "Report", 'Cleaning', 'Certification', 'Re-certification', 'Check-up',
                 'Re-enforce']  # ["Testings", "Cleaning"]
LIST_OF_ENDING_SEPARATORS = [' and ', ' & ', ', ']
LIST_OF_SEPARATORS = [' ', ' - ', ': ', ' on ', None]
'''LIST_OF_SERVICE_PADDINGS = ['Service', 'services', 'servicing', 'Support', 'support', 'Maintenance', 'Management',
                            'Planned Maintenance',
                            'Routine Maintenance', 'Routine Checkup', 'checkups', 'Check up']'''
LIST_OF_SERVICE_PADDINGS = ['NOT MANAGED BY PFM', 'UNITS decommissioned', 'SCHOOL MANAGING SERVICE',
                            'EQUIPMENT REMOVED OFF SITE',
                            'NOT REQUIRED UNDER PS TAKE OVER', 'NOT A PFM SITE', 'SITE CLOSED', "Inactive PM Request",
                            'Active Request By PES8910', 'REMOVED BMW', 'ADDED BMW', 'MODIFIED BMW',
                            'Changes/update due to Compliance FAILURE', 'Closed Only Required Short Term',
                            'SUBCONTRATOR ADVISED ASSET MOVED OFF SITE',
                            'NEW CHILD WO due to VENDOR changes', 'INACTIVE NO ASSETS ONSITE',
                            'INACTIVE NO LONGER MANAGER BY PFM',
                            'MERGED SITES WITH Westminster PS', 'MERGED SITES WITH Department of Education South PS',
                            'MERGED SITES WITH Department of Education North', 'DOE LEASE CEASED 2021',
                            'DOE LEASE CEASED from 2022',
                            'DOE-N LEASE CEASED IN 2023', 'DOE South LEASE CEASED IN 2023', 'NO ASSET Found',
                            'Site has NO ASSET', 'LEASE CEASED 31/01/20', 'LEASE commenced 06/11/2021',
                            'CEASING Lease on 23/05/2023',
                            'INVALID SITE ASSET/ASSET ID']
LIST_OF_MAINTENANCE_FREQS = ['1Y', 'Annually', 'yearly', '6M', 'weekly', 'daily', 'monthly', 'every month',
                             'Every 5 years', 'Every fortnight',
                             'Fortnightly', 'Every 2 weeks', 'every two weeks', 'once a week', 'regularly',
                             'On a daily basis', '2Y', '5Y',
                             'Every 2 years', 'every second year', 'every 2nd yr', 'every 5th yr', 'Every 5th Year',
                             'Every 3 months', 'every three months',
                             '3M', 'every weekends', 'As per Australian Standard', 'As per DOE request',
                             'as requested by Department of Education',
                             'In accordance with the DOE regulation',
                             'In accordance with Department of Education regulations', 'Once every three weeks',
                             'once every 3 week', 'Only when requested', 'upon requested']
HARMONISED_CLASS = 'Water Sampling Potable'
START_INDEX = 2897470
MAX_NUM_SAMPLES = 400
REQUEST_URL = 'https://prod-01.australiasoutheast.logic.azure.com:443/workflows/0ed70efeabef4468b16ebedd54c62111/triggers/manual/paths/invoke?api-version=2016-06-01&sp=%2Ftriggers%2Fmanual%2Frun&sv=1.0&sig=EwuXI_pguso2LmHjigGYRSOj1z0cNp_oGFUvsrOiGGI'
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
DEVICE = 'cpu'
NLP_AUGMENTERS = [
    na_sen.ContextualWordEmbsForSentenceAug(device=DEVICE),
    # na_word.BackTranslationAug(device=DEVICE),
    na_word.SynonymAug(),
    na_word.AntonymAug(),
    na_word.RandomWordAug(),
    # na_word.WordEmbsAug(model_type='glove'),
    na_word.SplitAug(name='Split_Aug', ),
    na_word.SpellingAug(),
    na_char.KeyboardAug(include_numeric=True),
    na_sen.AbstSummAug(device=DEVICE),
    na_sen.RandomSentAug(),
    nawcwe.ContextualWordEmbsAug(device=DEVICE),
    na_word.WordAugmenter(action='insert', device=DEVICE),
]


def generate_permutaions(list_of_strs: list) -> list:
    list_of_permutations = []
    for ending_separator in LIST_OF_ENDING_SEPARATORS:
        list_of_permutations.append([', '.join(permutation[:-1]) + ending_separator + permutation[-1] for permutation in
                                     list(itertools.permutations(list_of_strs))])
    return util_functions.flatten_list(list_of_permutations)


def generate_advanced_permutations(list_of_strs: list, separators=None) -> list:
    if separators is None:
        separators = [' ', ': ', ' - ', '_', '. ', '; ', None]
    str_permutation_list = list(itertools.permutations(list_of_strs))
    separator_permutaion_list = list(itertools.permutations(separators, len(list_of_strs) - 1))
    all_permutations = []
    for separator_permutaions in separator_permutaion_list:
        separator_permutaions = list(separator_permutaions)
        for str_permutations in str_permutation_list:
            str_permutations = list(str_permutations)
            new_str = ''
            # print(str_permutations)
            # print(separator_permutaions)
            for i in range(len(str_permutations)):  # individual string in this current permutation
                if i >= len(str_permutations) - 1:
                    new_str += str_permutations[i]
                else:
                    new_str = f'{new_str} ({str_permutations[i]}) ' if separator_permutaions[
                                                                           i] is None else f'{new_str}{str_permutations[i]}{separator_permutaions[i]}'
            all_permutations.append(new_str)

    return list(set(all_permutations))  # list(set(all_permutations))


def generate_optional_permutaions(list_of_strs: list, unified_separators=None) -> list:
    if unified_separators is None:
        unified_separators = [', ', '/', ' ']
    list_of_permutations = []
    for separator in unified_separators:
        list_of_permutations.append(
            [f'{separator}'.join(permutation) for permutation in list(itertools.permutations(list_of_strs))])
    return util_functions.flatten_list(list_of_permutations)


def generate_arbiratry_strings(arbitrary_str: str, main_subjects=None, seperators=None, reverse=False) -> list:
    # _separators = [' ', '-', ': ', ' or ', ['(', ')']
    arbitratry_strings = []

    if seperators is None:
        seperators = [' ', '-', ': ', ' or ', ['(', ')'], '/']

    if main_subjects is None:
        main_subjects = LIST_OF_MAIN_SUBJECTS

    for subject in main_subjects:
        for _separator in seperators:
            if type(_separator) is list:
                arbitratry_strings.append(f'{subject} {_separator[0]}{arbitrary_str}{_separator[1]}')
                if reverse:
                    arbitratry_strings.append(f'{arbitrary_str} {_separator[0]}{subject}{_separator[1]}')
            else:
                arbitratry_strings.append(f'{subject}{_separator}{arbitrary_str}')
                if reverse:
                    arbitratry_strings.append(f'{arbitrary_str}{_separator}{subject}')
    return arbitratry_strings


def get_place_permutation(list_of_permutations: list, main_subjects=None, separators=None, reverse=False) -> list:
    if main_subjects is None:
        main_subjects = LIST_OF_MAIN_SUBJECTS
    if separators is None:
        separators = LIST_OF_SEPARATORS
    place_permutations = []
    for subject in main_subjects:
        for seperator in separators:
            if seperator is None:
                place_permutations.append([f'{subject} ({permutation})' for permutation in list_of_permutations])
                if reverse:
                    place_permutations.append([f'{permutation} ({subject})' for permutation in list_of_permutations])
            else:
                place_permutations.append(
                    [f'{subject}{seperator}{permutation}' for permutation in list_of_permutations])
                if reverse:
                    place_permutations.append(
                        [f'{permutation}{seperator}{subject}' for permutation in list_of_permutations])
    return util_functions.flatten_list(place_permutations)


def _write_data_to_excel(excel_data: list, append=False):
    print(f'Last index is {START_INDEX + len(excel_data)}')
    if append:
        return util_functions.append_excel_workbook(
            file_path=SAVED_FILE_PATH,
            rows=[[i + START_INDEX + 1, excel_data[i], HARMONISED_CLASS] for i
                  in range(len(excel_data))],
            worksheet_name='Sheet')
    util_functions.save_dict_to_excel_workbook_with_row_formatting(
        rows=[[i + START_INDEX + 1, excel_data[i], HARMONISED_CLASS] for i in range(len(excel_data))],
        file_path=SAVED_FILE_PATH,
        headers=['idx', 'curr_desc', 'harmonised_desc'])


def retrieve_excel_info() -> tuple:
    excel_data = [{'idx': data['idx'],
                   'curr_desc': data['curr_desc'],
                   'harmonised_desc': data['harmonised_desc']}
                  for data in sorted(util_functions.read_excel_file(path=OG_EXCEL_PATH), key=itemgetter('idx'))]
    max_idx = excel_data[-1]['idx']
    grouped_data = []
    print('Grouping excel data:')
    for key, value in tqdm(
            groupby(sorted(excel_data, key=itemgetter('harmonised_desc')), lambda x: x['harmonised_desc'])):
        value = list(value)
        grouped_data.append({
            'harmonised_desc': key,
            'samples': value,
            'num_samples': len(value),
        })
    return grouped_data, max_idx


def analyse_excel_info(excel_data_infos: list, min_set_size=None) -> list:
    if min_set_size is None:
        min_set_size = 1250
    excel_data_infos.sort(key=itemgetter('num_samples'))
    for info_data in excel_data_infos:
        info_data['all_curr_trimmed_descs'] = [util_functions.lower_case_and_clear_white_space(row['curr_desc']) for row
                                               in info_data['samples']]
        print(f"{info_data['harmonised_desc']} has {info_data['num_samples']} samples")
    return [data for data in excel_data_infos if data['num_samples'] < min_set_size]


def retrieve_small_excel_dataset() -> list:
    excel_data = [data for data in util_functions.read_excel_file(path=SAVED_SMALL_SAMPLE_FILE_PATH) if
                  data['harmonised_desc'] == HARMONISED_CLASS]
    for data in excel_data:
        data['curr_trimmed_descs'] = util_functions.lower_case_and_clear_white_space(data['curr_desc'])
    return excel_data


def _is_service_padding_already_included(service_padding: str, all_curr_trimmed_descs: list) -> bool:
    for curr_trimmed_descs in all_curr_trimmed_descs:
        if util_functions.are_strings_similar(curr_trimmed_descs, service_padding, to_regex=True):
            return True
    return False


def generate_all_possible_terms(list_of_main_subjs: list) -> list:
    return list(set(
        util_functions.flatten_list([
            list_of_main_subjs,
            [f'{main_subj}.' for main_subj in list_of_main_subjs],
            [f'{main_subj}:' for main_subj in list_of_main_subjs]
        ])
    ))
    _all_possible_terms = []
    all_permutations = []
    _sp_words = generate_permutaions(LIST_OF_WORDS)
    for main_subject in list_of_main_subjs:
        for freq in LIST_OF_MAINTENANCE_FREQS:
            for sp_word in _sp_words:
                all_permutations.append(
                    generate_advanced_permutations([main_subject, sp_word, freq], separators=[' ', ' - ', ': ']))

    _all_possible_terms = list(set(
        util_functions.flatten_list(all_permutations),
    ))
    util_functions.random_seed_shuffle(seed=int(START_INDEX / 200), og_list=_all_possible_terms)

    return _all_possible_terms if len(_all_possible_terms) <= MAX_NUM_SAMPLES else _all_possible_terms[
                                                                                   0:MAX_NUM_SAMPLES]


# _write_data_to_excel(excel_data=_all_possible_terms, append=APPEND_TO_EXCEL)


def generate_new_main_sub_str(main_sub_str: str) -> list:
    def remove_suffixes_and_word(string: str) -> str:
        # Remove 's and s at the end of each word
        cleaned_string = re.sub(r"(\w+)('s|s)\b", r"\1", string)
        # Remove the standalone word "and"
        cleaned_string = re.sub(r"\band\b", "", cleaned_string)
        return cleaned_string

    return util_functions.flatten_list([
        [main_sub_str],
        [f'{main_sub_str}:', f':{main_sub_str}:', f':{main_sub_str}'],
        [f'{main_sub_str} -', f'- {main_sub_str} -', f'- {main_sub_str}'],
        [remove_suffixes_and_word(main_sub_str), util_functions.str_to_regex(main_sub_str),
         util_functions.str_to_regex(remove_suffixes_and_word(main_sub_str))]
    ])


def perform_nlp_aug(main_subjects: list, num_gens=4) -> list:
    augmented_main_subjs = set()
    for main_subject in set(main_subjects):
        for augmenter in NLP_AUGMENTERS:
            [augmented_main_subjs.add(augmented_str) for augmented_str in
             set(augmenter.augment(main_subject, n=num_gens))]

    augmented_main_subjs = list(set(augmented_main_subjs))
    augmented_main_subjs.extend(main_subjects)
    return augmented_main_subjs


def main():
    upload_json_datas = []
    idx = START_INDEX
    saved_json_permutaions = util_functions.read_from_json_file(JSON_FILE_PATH)
    saved_json_permutaions = sorted(saved_json_permutaions, key=itemgetter('class_idx'))
    print('Perform initial data augmentation')
    '''for saved_json_permutaion in tqdm(saved_json_permutaions):
        saved_json_permutaion['main_subjs'] = perform_nlp_aug(saved_json_permutaion['main_subjs'])'''

    for item in tqdm(saved_json_permutaions):
        list_of_word_copied = copy.copy(LIST_OF_WORDS)
        random.shuffle(list_of_word_copied)
        list_of_word_copied = list_of_word_copied[:3]

        curr_desc_w_service_paddings = util_functions.flatten_list([
            generate_arbiratry_strings(arbitrary_str=service_padding, main_subjects=item['main_subjs'],
                                       seperators=[' ', ': ', ' - '], reverse=True)
            for service_padding in LIST_OF_SERVICE_PADDINGS])
        curr_desc_w_freqs = get_place_permutation(list_of_permutations=LIST_OF_MAINTENANCE_FREQS,
                                                  main_subjects=item['main_subjs'],
                                                  reverse=True)
        curr_desc_w_sp_words = get_place_permutation(list_of_permutations=generate_permutaions(list_of_word_copied),
                                                     main_subjects=item['main_subjs'],
                                                     reverse=True)
        for new_list in [curr_desc_w_freqs, curr_desc_w_sp_words, curr_desc_w_service_paddings]:
            util_functions.random_seed_shuffle(seed=len(new_list), og_list=new_list)

        all_permutations = set()
        for main_subject in set(item['main_subjs']):
            list_of_word_copied = copy.copy(LIST_OF_WORDS)
            util_functions.random_seed_shuffle(seed=int(len(main_subject)), og_list=list_of_word_copied)
            list_of_word_copied = list_of_word_copied[:3]
            list_of_words_permutation = generate_permutaions(list_of_word_copied)

            for sp_word in set(list_of_words_permutation):
                random_frequencies = copy.copy(LIST_OF_MAINTENANCE_FREQS)
                util_functions.random_seed_shuffle(seed=item['class_idx'], og_list=random_frequencies)
                random_frequencies = random_frequencies[:math.ceil(len(random_frequencies) / 1.5)]

                random.shuffle(LIST_OF_SERVICE_PADDINGS)
                for freq in set(random_frequencies):
                    [all_permutations.add(generated_str) for generated_str in
                     generate_advanced_permutations(list_of_strs=[freq, main_subject, sp_word],
                                                    separators=[' ', ' - '])]
                for service_padding in set(LIST_OF_SERVICE_PADDINGS):
                    [all_permutations.add(generated_str) for generated_str in
                     generate_advanced_permutations(list_of_strs=[service_padding, main_subject, sp_word],
                                                    separators=[' - '])]
                    for freq in set(random_frequencies[:math.ceil(len(random_frequencies) / 10)]):
                        [all_permutations.add(generated_str) for generated_str in
                         generate_advanced_permutations(list_of_strs=[freq, main_subject, service_padding, sp_word],
                                                        separators=[' ', ': ', ' - '])]

        all_possible_terms = perform_nlp_aug(
            main_subjects=list(set(util_functions.flatten_list([
                util_functions.flatten_list(list(all_permutations)),
                curr_desc_w_freqs,
                curr_desc_w_sp_words,
                item['main_subjs'],
            ]))), num_gens=2)

        util_functions.random_seed_shuffle(seed=10, og_list=all_possible_terms)
        for main_subject in item['main_subjs']:
            all_possible_terms.insert(0, main_subject)

        item['upload_data'] = []
        for possible_term in set(all_possible_terms) if len(all_possible_terms) <= MAX_NUM_SAMPLES else set(
                all_possible_terms[
                0:MAX_NUM_SAMPLES]):
            item['upload_data'].append(
                {'idx': idx, 'class_uid': item['class_uid'], 'harmonised_desc': item['class_label'],
                 'curr_desc': possible_term})
            idx += 1
        '''for term in util_functions.flatten_list(
                [generate_new_main_sub_str(main_subj) for main_subj in item['main_subjs']]):
            item['upload_data'].append(
                {'idx': idx, 'class_uid': item['class_uid'], 'harmonised_desc': item['class_label'],
                 'curr_desc': term})
            idx += 1'''

        upload_json_datas.append(item['upload_data'])
        if len(item['upload_data']) < MAX_NUM_SAMPLES:
            print(f"{item['class_label']} has {len(item['upload_data'])} items\n")

    upload_json_datas = util_functions.flatten_list(upload_json_datas)

    _map_iter = 3
    curr_row_counts = 0
    step = 1048570
    start = 0
    end = step

    print('Uploading:')
    for i in tqdm(range(0, len(upload_json_datas), step)):
        # response = requests.post(REQUEST_URL, json=upload_json_datas[start:end])
        rows = [
            [
                upload_json_data['idx'],
                upload_json_data['curr_desc'],
                upload_json_data['class_uid'],
                upload_json_data['harmonised_desc']
            ] for upload_json_data in upload_json_datas[start:end]
        ]
        util_functions.save_dict_to_excel_workbook_with_row_formatting(
            file_path=f'{SAVED_FILE_PATH}{_map_iter}.xlsx',
            headers=WORKBOOK_HEADERS,
            rows=rows)

        start = end
        end += step
        _map_iter += 1

    print(f'Last index is {idx}')


if __name__ == '__main__':
    main()
