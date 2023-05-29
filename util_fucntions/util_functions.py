import itertools
import math, re, json, csv, os, openpyxl
import pandas as pd
from operator import itemgetter
from itertools import groupby, filterfalse
from pandas.io.json import json_normalize
from datetime import datetime
from tqdm import tqdm
import random


def subtract_lists(large_list: list, small_list: list) -> list:
    return [item for item in large_list if item not in small_list]


def flatten_list(multi_dim_list: list) -> list:
    return list(itertools.chain.from_iterable(multi_dim_list))


def reformat_key(old_key: str) -> str:
    return old_key.lower().replace(' ', '_')


def clean_white_space(old_str: str) -> str:
    return re.sub(r'\s+', ' ', old_str)


def remove_brackets(old_str: str):
    return re.sub(r'\([^()]*\)', '', old_str)


def lower_case_and_clear_white_space(og_string: str, to_regex=False) -> str:
    new_string = og_string.lower().replace(" ", '')
    return re.sub(r'\W+', '', new_string) if to_regex else new_string


def are_strings_the_same(str1: str, str2: str, to_regex=False) -> bool:
    return lower_case_and_clear_white_space(str2, to_regex) == lower_case_and_clear_white_space(str1, to_regex)


def are_strings_similar(str1: str, str2: str, to_regex=False) -> bool:
    str1 = lower_case_and_clear_white_space(str1, to_regex)
    str2 = lower_case_and_clear_white_space(str2, to_regex)
    return str1 in str2 or str2 in str1


def format_dictionaries(dict_list: list) -> list:
    for data in dict_list:
        modified_keys = {}
        for key, value in data.items():
            modified_keys[reformat_key(key)] = value
        data.clear()
        data.update(modified_keys)
    return dict_list


def dictionary_has_nan(dictionary: dict) -> bool:
    for value in dictionary.values():
        if isinstance(value, float) and math.isnan(value):
            return True
    return False


def is_nan(value) -> bool:
    if type(value) != float or type(value) != int:
        return value == 'nan'
    return math.isnan(value)


def group_dict_list_by_key(dict_list: list, key_to_grp: str, grouped_value_key='grouped_vals'):
    _grouped_data = []
    for key, value in groupby(sorted(dict_list, key=itemgetter(key_to_grp)), lambda x: x[key_to_grp]):
        _grouped_data.append({
            key_to_grp: key,
            grouped_value_key: list(value)
        })
    return _grouped_data


def read_workbook(path: str, format_key=False) -> list:
    df = pd.read_excel(path)
    df = df.replace(u'\xa0', ' ', regex=True)
    print(f'{path}: ')
    read_data = [x for x in tqdm(df.to_dict(orient="records")) if not pd.isna(x)]
    return read_data if not format_key else format_dictionaries(read_data)


def read_csv(path: str, format_key=False) -> list:
    read_data = []
    print(f'{path}: ')
    with open(path, 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in tqdm(reader):
            read_data.append(row)

    # read_data = json.dumps(read_data)
    return read_data if not format_key else format_dictionaries(read_data)


def read_excel_file(path: str, format_key=False) -> list:
    if path.isspace():
        return []
    print('Reading from', end=': ')
    if os.path.splitext(path)[1] == '.csv':
        print('csv file')
        return read_csv(path=path, format_key=format_key)
    elif os.path.splitext(path)[1] == '.xlsx':
        print('excel workbook file')
        return read_workbook(path=path, format_key=format_key)
    else:
        print('nothing')
        return []


def is_json(data):
    try:
        json.loads(data)
        return True
    except ValueError:
        return False


def save_dict_to_excel_workbook_with_row_formatting(file_path: str, headers: list, rows: list) -> None:
    if file_path.isspace():
        return
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.append(headers)
    print('Saving to excel...')
    for row in tqdm(rows):
        sheet.append(row)
    workbook.save(file_path)


def convert_datetime_obj_to_str(datetime_obj: datetime, str_format='%Y-%m-%d %H:%M:%S') -> str:
    return datetime_obj.strftime(str_format)


def random_seed_shuffle(seed: int, og_list: list) -> None:
    random.seed(seed)
    random.shuffle(og_list)
