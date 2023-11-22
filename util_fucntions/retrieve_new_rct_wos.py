from util_fucntions import util_functions
from tqdm import tqdm
from itertools import groupby
from operator import getitem

ALL_RCT_WO_PATH = '/Users/naga/Library/Mobile Documents/com~apple~CloudDocs/PFM stuffs/DCA sample/ReactiveWorkOrderReviews.xlsx'
SUBSET_RCT_PATH = '/Users/naga/Library/Mobile Documents/com~apple~CloudDocs/PFM stuffs/DCA sample/DCA RCT Asset Data.csv'
SAVED_PATH = '/Users/naga/Library/Mobile Documents/com~apple~CloudDocs/PFM stuffs/DCA sample/dca_test_samples.xlsx'
HEADERS = ['Work Order Number', 'Description', 'Completion Notes']


def main():
    all_rct_wo_data = util_functions.read_excel_file(path=ALL_RCT_WO_PATH, format_key=True)
    subset_wo_ids = [util_functions.lower_case_and_clear_white_space(data['work_order_number']) for data in
                     util_functions.read_excel_file(path=SUBSET_RCT_PATH, format_key=True)]

    util_functions.save_dict_to_excel_workbook_with_row_formatting(file_path=SAVED_PATH,
                                                                   headers=HEADERS,
                                                                   rows=[
                                                                       [data['work_order'], data['work_description'], data['completion_notes']]
                                                                       for data in all_rct_wo_data if
                                                                       util_functions.lower_case_and_clear_white_space(
                                                                           data['work_order']) not in subset_wo_ids
                                                                   ])


if __name__ == '__main__':
    main()
