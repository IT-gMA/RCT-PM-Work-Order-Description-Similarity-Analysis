import util_functions
from tqdm import tqdm
from operator import itemgetter
from datetime import datetime

DESC_MAPPING_FILE_PATH = '../xlsx_resources/for_trainings/rct_pm_desc_similarity.xlsx'
RCT_WORK_ORDER_FILE_PATH = '../../xlsx_resources/work_orders/RCT_work_order.xlsx'


def _wo_completion_date_to_dt_obj(completion_date: str, wo_num='') -> datetime:
    # print(f'wo_num: {wo_num} - {completion_date}')
    return datetime.strptime(completion_date, "%Y-%m-%d %H:%M:%S")


def _wo_completion_date_to_dt_obj_alt1(completion_date: str, wo_num='') -> datetime:
    # print(f'wo_num: {wo_num} - {completion_date}')
    return datetime.strptime(completion_date, "%d/%m/%Y %H:%M:%S")


def _wo_created_on_date_to_dt_obj(created_on: str, wo_num='') -> datetime:
    # print(f'wo_num: {wo_num} - {created_on}')
    return datetime.strptime(created_on, "%d/%m/%Y")


def _format_rct_work_order_fields(worker_orders: list, no_asset_wos=None) -> list:
    if no_asset_wos is None:
        no_asset_wos = []
    return [{
        'wo_num': wo['wo_#'],
        'wo_desc': wo['description'],
        'contract': wo['agency'],
        'priority': wo['priority'],
        'location': wo['location_full_desc.'],
        'category': wo['category_of_works'],
        'trade_grp': wo['trade_group'],
        'trade': wo['trade'],
        'vendor': wo['vendor_name'],
        'status': wo['status'],
        'comp_status': wo['comp_status'],
    } for wo in worker_orders if wo['wo_#'] != 'nan' and not wo['wo_#'].isspace()]


def read_rct_work_orders(no_asset_wos=None) -> list:
    return _format_rct_work_order_fields(util_functions.read_excel_file(path=RCT_WORK_ORDER_FILE_PATH, format_key=True),
                                         no_asset_wos=no_asset_wos)


def read_rct_pm_desc_map_data() -> list:
    return util_functions.read_excel_file(DESC_MAPPING_FILE_PATH)


def _assign_trade_as_context(rct_wos: list, desc_wo_maps: list) -> None:
    _map_file_path = '../../xlsx_resources/book88.xlsx'
    first = True
    print('Start mapping:')
    for desc_wo_map in tqdm(desc_wo_maps):
        wo_id = util_functions.clean_white_space(desc_wo_map['mapping_code'].split(':')[0])
        rct_wo = [rct for rct in rct_wos if util_functions.clean_white_space(rct['wo_num']) == wo_id][0]

        _saved_row = [
            [desc_wo_map['mapping_code'], f"{rct_wo['trade_grp']} {rct_wo['trade']} : {desc_wo_map['rct_desc']}", desc_wo_map['pm_wo_desc'], desc_wo_map['similarity']]
        ]
        if first:
            first = False
            util_functions.save_dict_to_excel_workbook_with_row_formatting(file_path=_map_file_path,
                                                                           headers=['mapping_code', 'rct_desc', 'pm_wo_desc', 'similarity'],
                                                                           rows=_saved_row)
        else:
            util_functions.append_excel_workbook(file_path=_map_file_path, rows=_saved_row)


def main():
    rct_wos = read_rct_work_orders()
    desc_wo_maps = read_rct_pm_desc_map_data()
    _assign_trade_as_context(rct_wos=rct_wos, desc_wo_maps=desc_wo_maps)


if __name__ == '__main__':
    main()
