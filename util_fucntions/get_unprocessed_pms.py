from util_fucntions import util_functions
from tqdm import tqdm
from itertools import groupby
from operator import getitem

PROCESSED_PM_PATHS = [
    '/Users/naga/Library/Mobile Documents/com~apple~CloudDocs/PFM stuffs/GAP app/test_web_doc/csv_resources/PMLocationMaps.csv',
    '/Users/naga/Library/Mobile Documents/com~apple~CloudDocs/PFM stuffs/GAP app/test_web_doc/csv_resources/PMLocationMaps2.csv']

ALL_PM_PATHS = [
    '/Users/naga/Library/Mobile Documents/com~apple~CloudDocs/PFM stuffs/GAP app/test_web_doc/csv_resources/PMLocationMapsv3.csv']

NEW_PM_PATH = '/Users/naga/Downloads/Book6.xlsx'


def main():
    processed_pms = []
    all_pms = []
    non_dupl_pm_codes = []
    for path in PROCESSED_PM_PATHS:
        processed_pms.append(util_functions.read_excel_file(path=path, format_key=True))
    for path in ALL_PM_PATHS:
        all_pms.append(util_functions.read_excel_file(path=path, format_key=True))
    all_pms = util_functions.flatten_list(all_pms)
    '''for data in all_pms:
        if data['pm_code'] in non_dupl_pm_codes:
            print(f'Duplicate: {data}')
        else:
            non_dupl_pm_codes.append(data['pm_code'])'''

    processed_pms = list(set([f"{data['pm_code']}###{data['location_code']}###{data['pm_status']}" for data in
                              util_functions.flatten_list(processed_pms)]))

    all_pms = [data for data in all_pms if
               f"{data['pm_code']}###{data['location_code']}###{data['pm_status']}" not in processed_pms]
    util_functions.save_dict_to_excel_workbook_with_row_formatting(file_path=NEW_PM_PATH,
                                                                   headers=['Site', 'PM Code', 'PM Description',
                                                                            'Location', 'Location Code', 'PM Status', 'Reported Date'],
                                                                   rows=[
                                                                       [data["\ufeffsite"], data['pm_code'], data['pm_description'], data['location'], data['location_code'], data['pm_status'], data['reported_date']]
                                                                       for data in all_pms])
    return


if __name__ == '__main__':
    main()
