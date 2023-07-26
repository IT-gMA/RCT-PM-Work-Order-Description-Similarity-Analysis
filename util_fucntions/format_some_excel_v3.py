from util_fucntions import util_functions
from tqdm import tqdm
import itertools
from itertools import groupby
from operator import itemgetter
import requests

OG_EXCEL_PATH = '../xlsx_resources/for_trainings/maximo_pm_to_gap_pm_desc_map.xlsx'
SAVED_FILE_PATH = '../xlsx_resources/for_trainings/maximo_pm_desc_map_reworked_XS/maximo_pm_to_gap_pm_desc_map_it'
SAVED_SMALL_SAMPLE_FILE_PATH = '/Users/naga/Downloads/small_sample_maximo_pm_to_gap_pm_desc_map.xlsx'
WORKBOOK_HEADERS = ['idx', 'curr_desc', 'harmonised_desc']
APPEND_TO_EXCEL = True
LIST_OF_WORDS = ["Testing", 'Re-certification', "Cleaning"]  # ["Testings", "Cleaning"]
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
                            'Site has NO ASSET', 'LEASE CEASED 31/01/20', 'LEASE commenced 06/11/2021', 'CEASING Lease on 23/05/2023',
                            'INVALID SITE ASSET/ASSET ID']
LIST_OF_MAINTENANCE_FREQS = ['1Y', 'Annually', 'yearly', '6M', 'weekly', 'daily', 'monthly', 'every month',
                             'Every 5 years', 'Every fortnight', 'Once EVERY 2 WEEKS',
                             'as requested by Department of Education', 'ONE OFF SERVICE']
HARMONISED_CLASS = 'Water Sampling Potable'
print(f'CLass label is {HARMONISED_CLASS}')
START_INDEX = 1697502
MAX_NUM_SAMPLES = 145
REQUEST_URI = "https://prod-01.australiasoutheast.logic.azure.com:443/workflows/3fafd3aaca5f4fbc9d346a99db7e7d67/triggers/manual/paths/invoke?api-version=2016-06-01&sp=%2Ftriggers%2Fmanual%2Frun&sv=1.0&sig=VBHoh6Hoae9mzF1-WqF9COY5R9tcN6Y2L_iDYrkjKxU"


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

def main():
    upload_json_datas = []
    idx = START_INDEX
    '''saved_json_permutaions = util_functions.read_from_json_file(
        file_path='../rcnn_model_trainings/workorder_classifications/pm_harmonisation_classification/saved_permutaions.json')'''
    saved_json_permutaions = [
        {
            "class_idx": 23,
            "class_label": "Emergency Eye Wash / Safety Shower",
            "main_subjs": [
                'Eye Wash', 'EyeWash', 'Safty Showers', 'Safety Shower', 'SafetyShower', 'EmergencyEye-wash'
            ]
        },
    ]
    for item in tqdm(saved_json_permutaions):
        curr_desc_w_service_paddings = util_functions.flatten_list([
            generate_arbiratry_strings(arbitrary_str=service_padding, main_subjects=item['main_subjs'],
                                       seperators=[' ', ': ', ' - '], reverse=True)
            for service_padding in LIST_OF_SERVICE_PADDINGS])
        curr_desc_w_freqs = get_place_permutation(list_of_permutations=LIST_OF_MAINTENANCE_FREQS,
                                                  main_subjects=item['main_subjs'],
                                                  reverse=True)
        curr_desc_w_sp_words = get_place_permutation(list_of_permutations=generate_permutaions(LIST_OF_WORDS),
                                                     main_subjects=item['main_subjs'],
                                                     reverse=True)
        for new_list in [curr_desc_w_freqs, curr_desc_w_sp_words, curr_desc_w_service_paddings]:
            util_functions.random_seed_shuffle(seed=len(new_list), og_list=new_list)

        all_permutations = []
        for main_subject in item['main_subjs']:
            for freq in LIST_OF_MAINTENANCE_FREQS:
                for service_padding in LIST_OF_SERVICE_PADDINGS:
                    all_permutations.append(generate_advanced_permutations(list_of_strs=[main_subject, service_padding, freq], separators=[' ', ': ', ' - ', '. ', '-']))
                for sp_word in LIST_OF_WORDS:
                    all_permutations.append(generate_advanced_permutations(list_of_strs=[main_subject, sp_word, freq], separators=[' ', ': ', ' - ', '. ', '-']))

        all_possible_terms = list(set(util_functions.flatten_list([
            util_functions.flatten_list(all_permutations),
            #curr_desc_w_freqs[0:int(0.1 * len(curr_desc_w_freqs))],
            #curr_desc_w_sp_words[0:int(0.25 * len(curr_desc_w_sp_words))],
            #curr_desc_w_service_paddings[0:int(0.3 * len(curr_desc_w_service_paddings))],
            # [f'{main_subject}:' for main_subject in item['main_subjs']],
        ])))
        util_functions.random_seed_shuffle(seed=10, og_list=all_possible_terms)

        item['upload_data'] = []
        _max_quantity = MAX_NUM_SAMPLES
        if 'OWS' in item['class_label']:
            _max_quantity = 28
        elif 'FIP' in item['class_label']:
            _max_quantity = 70
        for possible_term in all_possible_terms if len(all_possible_terms) < _max_quantity else all_possible_terms[0:_max_quantity]:
            item['upload_data'].append({'idx': idx, 'harmonised_desc': item['class_label'], 'curr_desc': possible_term})
            idx += 1
        upload_json_datas.append(item['upload_data'])
        print(f"{item['class_label']} has {len(item['upload_data'])} items\n")

    # upload_json_datas = util_functions.flatten_list(upload_json_datas)
    saved_excel_data = []

    _map_iter = 4
    curr_row_counts = 0
    for i in range(0, len(upload_json_datas)):
        upload_json_data = upload_json_datas[i]
        curr_row_counts += len(upload_json_data)
        rows = [[row['idx'], row['curr_desc'], row['harmonised_desc']] for row in upload_json_data]

        if i < 1:
            util_functions.save_dict_to_excel_workbook_with_row_formatting(
                file_path=f'{SAVED_FILE_PATH}{_map_iter}.xlsx',
                headers=WORKBOOK_HEADERS,
                rows=rows)
        else:
            if curr_row_counts >= 1048572:
                curr_row_counts = 0
                _map_iter += 1
                util_functions.save_dict_to_excel_workbook_with_row_formatting(
                    file_path=f'{SAVED_FILE_PATH}{_map_iter}.xlsx',
                    headers=WORKBOOK_HEADERS,
                    rows=rows)
            else:
                util_functions.append_excel_workbook(file_path=f'{SAVED_FILE_PATH}{_map_iter}.xlsx',
                                                     rows=rows)
    print(f'Last index is {idx}')


if __name__ == '__main__':
    main()
