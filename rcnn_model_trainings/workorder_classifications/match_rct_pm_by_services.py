import sys, os, math
from util_fucntions import util_functions
from operator import itemgetter
from itertools import groupby, filterfalse
from datetime import datetime
from tqdm import tqdm
import random
import time

PM_WORK_ORDER_FILE_PATH = '../../xlsx_resources/work_orders/PM_work_order.xlsx'
RCT_WORK_ORDER_FILE_PATH = '../../xlsx_resources/work_orders/RCT_work_order.xlsx'
NO_ASSET_WORK_ORDER_FILE_PATH = '../../xlsx_resources/work_orders/no_asset_work_orders.xlsx'
SAVED_MAPPED_FILE_PATH = '../saved_resources/rct_pm_maps_on_contracts_alt1.xlsx'

MAPPED_WORK_ORDER_HEADERS = ['Work Order Number', 'Description', 'Contract Code', 'Priority', 'Location', 'Category',
                             'Trade Group', 'Trade',
                             'Vendor', 'Reported on', 'Target Completion Date', 'Actual Completion Date',
                             'Has PMs against', 'Number of PMs against',
                             'PM Work Order', 'PM Location', 'PM Description']
ACTIVE_PM_STATUSES = ['comp', 'compfin', 'close', 'fldcomp']


def _wo_completion_date_to_dt_obj(completion_date: str, wo_num='') -> datetime:
    # print(f'wo_num: {wo_num} - {completion_date}')
    return datetime.strptime(completion_date, "%Y-%m-%d %H:%M:%S")


def _wo_completion_date_to_dt_obj_alt1(completion_date: str, wo_num='') -> datetime:
    # print(f'wo_num: {wo_num} - {completion_date}')
    return datetime.strptime(completion_date, "%d/%m/%Y %H:%M:%S")


def _wo_created_on_date_to_dt_obj(created_on: str, wo_num='') -> datetime:
    # print(f'wo_num: {wo_num} - {created_on}')
    return datetime.strptime(created_on, "%d/%m/%Y")


def _wo_issued_on_date_to_dt_obj(created_on: str, wo_num='') -> datetime:
    # print(f'wo_num: {wo_num} - {created_on}')
    return datetime.strptime(created_on, "%d %b %Y")


def _filter_active_wo(work_order: dict, active_only: bool) -> bool:
    return util_functions.lower_case_and_clear_white_space(
        work_order['comp_status']) == 'pass' and util_functions.lower_case_and_clear_white_space(
        work_order['status']) in ACTIVE_PM_STATUSES if active_only else True


def _format_no_asset_work_orders(work_orders: list, filtered_categories=None) -> list:
    if filtered_categories is None:
        filtered_categories = []
    return [{
        'wo_num': wo['work_order_number'],
        'wo_desc': wo['description'],
        'contract': wo['contract_id'],
        'trade_grp': wo['trade_group'],
        'trade': wo['trade'],
        'vendor': wo['vendor_name'],
        'location': wo['location'],
        'category': wo['work_type_das'],
    } for wo in work_orders if wo['work_order_number'] != 'nan' and not wo['work_order_number'].isspace() and wo[
        'has_assets'].lower() == 'no' and (True if len(filtered_categories) < 1 else wo[
                                                                                         'work_type_das'].lower() in filtered_categories)]


def _format_pm_work_order_fields(worker_orders: list, active_only=False, no_asset_wos=None) -> list:
    if no_asset_wos is None:
        no_asset_wos = []

    return [{
        'wo_num': wo['wo_#'],
        'wo_desc': wo['description'],
        'contract': wo['agency'],
        'location': wo['location_full_desc.'],
        # 'category': wo['category_of_works'],
        # 'statutory': wo['statutory'],
        'trade_grp': wo['trade_group'],
        'trade': wo['trade'],
        'vendor': wo['vendor_name'],
        'status': wo['status'],
        # 'created_on': _wo_issued_on_date_to_dt_obj(wo['reported_date'], wo['wo_#']),
        'target_comp_date': _wo_completion_date_to_dt_obj(wo['target_comp'], wo['wo_#']),
        'actual_comp_date': _wo_completion_date_to_dt_obj(
            wo['actual_comp'] if wo['actual_comp'] == 'nan' else wo['target_comp'], wo['wo_#']),
        'comp_status': wo['comp_status'],
        # 'comp_notes': wo['completion_notes']
    } for wo in worker_orders if wo['wo_#'] != 'nan' and not wo['wo_#'].isspace()
                                 and _filter_active_wo(wo, active_only)
                                 and (
                                     True if len(no_asset_wos) < 1 else util_functions.lower_case_and_clear_white_space(
                                         wo['wo_#']) not in no_asset_wos)]


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
        'created_on': _wo_issued_on_date_to_dt_obj(wo['reported_date']),
        'target_comp_date': _wo_completion_date_to_dt_obj(wo['target_comp'], wo['wo_#']),
        'actual_comp_date': _wo_completion_date_to_dt_obj(
            wo['actual_comp'] if wo['actual_comp'] == 'nan' else wo['target_comp'], wo['wo_#']),
        'comp_status': wo['comp_status'],
        # 'comp_notes': wo['completion_notes']
    } for wo in worker_orders if wo['wo_#'] != 'nan' and not wo['wo_#'].isspace()
                                 and (
                                     True if len(no_asset_wos) < 1 else util_functions.lower_case_and_clear_white_space(
                                         wo['wo_#']) not in no_asset_wos)]


def _format_rct_pm_data_for_workbook(rct_wos: list, minimise=0) -> list:
    new_data = []
    for rct in tqdm(sorted(rct_wos, key=itemgetter('created_on'))):
        if rct['num_pms'] < 1:
            new_data.append([rct['wo_num'], rct['wo_desc'], rct['contract'],
                             rct['priority'], rct['location'], rct['category'],
                             rct['trade_grp'], rct['trade'], rct['vendor'],
                             util_functions.convert_datetime_obj_to_str(rct['created_on']),
                             util_functions.convert_datetime_obj_to_str(rct['target_comp_date']),
                             util_functions.convert_datetime_obj_to_str(rct['actual_comp_date']), False, rct['num_pms'],
                             None, None, None
                             ])
            continue
        for idx in range(0, minimise if minimise > 0 else len(rct['pms'])):
            try:
                _pm = rct['pms'][idx]
            except IndexError:
                continue

            if idx == 0:
                new_data.append([rct['wo_num'], rct['wo_desc'], rct['contract'],
                                 rct['priority'], rct['location'], rct['category'],
                                 rct['trade_grp'], rct['trade'], rct['vendor'],
                                 util_functions.convert_datetime_obj_to_str(rct['created_on']),
                                 util_functions.convert_datetime_obj_to_str(rct['target_comp_date']),
                                 util_functions.convert_datetime_obj_to_str(rct['actual_comp_date']), True,
                                 rct['num_pms'],
                                 _pm['wo_num'], _pm['location'], _pm['wo_desc']
                                 ])
            else:
                new_data.append([None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                                 _pm['wo_num'], _pm['location'], _pm['wo_desc']
                                 ])
    return new_data


def read_pm_work_orders(active_only=False, no_asset_wos=None) -> list:
    return _format_pm_work_order_fields(util_functions.read_excel_file(path=PM_WORK_ORDER_FILE_PATH, format_key=True),
                                        active_only=active_only, no_asset_wos=no_asset_wos)


def read_rct_work_orders(no_asset_wos=None) -> list:
    return _format_rct_work_order_fields(util_functions.read_excel_file(path=RCT_WORK_ORDER_FILE_PATH, format_key=True),
                                         no_asset_wos=no_asset_wos)


def read_work_order_data(wo_path: str) -> list:
    return _format_pm_work_order_fields(util_functions.read_excel_file(path=wo_path, format_key=True))


def read_no_asset_work_orders(filtered_categories=None) -> list:
    return _format_no_asset_work_orders(
        work_orders=util_functions.read_excel_file(path=NO_ASSET_WORK_ORDER_FILE_PATH, format_key=True),
        filtered_categories=filtered_categories)


def _map_rct_to_pms(rct_wos: list, pm_wos: list) -> tuple:
    _rct_with_pms = []
    _rct_without_pms = []
    print('Mapping RCT to PM Work Orders:')
    for rct_wo in tqdm(rct_wos):
        rct_wo['pms'] = [pm_wo for pm_wo in pm_wos if
                         # _filter_active_wo(work_order=pm_wo, active_only=True) and
                         util_functions.are_strings_the_same(pm_wo['trade_grp'], rct_wo['trade_grp'])
                         and util_functions.are_strings_the_same(pm_wo['trade'], rct_wo['trade'])
                         # nd pm_wo['actual_comp_date'] < rct_wo['created_on']
                         # and util_functions.are_strings_the_same(pm_wo['contract'], rct_wo['contract'])
                         # and util_functions.are_strings_the_same(pm_wo['location'], rct_wo['location'])
                         ]
        rct_wo['has_pms'] = len(rct_wo['pms']) > 0
        rct_wo['num_pms'] = len(rct_wo['pms'])
        if rct_wo['has_pms']:
            _rct_with_pms.append(rct_wo)
        else:
            _rct_without_pms.append(rct_wo)
    return _rct_with_pms, _rct_without_pms


def map_rct_desc_to_pm_desc(pm_wos: list, rct_wos: list) -> None:
    _map_iter = 0
    _map_file_path = '../../xlsx_resources/for_trainings/rct_pm_desc_similarity_v2'
    # copy_rct_wos = sorted(rct_wos, key=itemgetter('wo_desc'))
    print('Map RCT to PM work order descriptions:')
    no_duplicate_pm_desc = sorted(list(set([pm['wo_desc'] for pm in pm_wos])))

    rct_wo_idx = 0
    for rct_wo in rct_wos:
        curr_rows = []
        assigned_pm_descs = [pm['wo_desc'] for pm in rct_wo['pms']]
        candidate_pm_descs = [pm_desc for pm_desc in no_duplicate_pm_desc if pm_desc not in assigned_pm_descs]
        util_functions.random_seed_shuffle(seed=rct_wo_idx + 1, og_list=candidate_pm_descs)
        candidate_pm_descs = util_functions.flatten_list([
            assigned_pm_descs,
            candidate_pm_descs[0:3] if len(candidate_pm_descs) > 3 else candidate_pm_descs
        ])

        print(f"{round(rct_wo_idx * 100 / len(rct_wos), 2)}% -- reading from {rct_wo['wo_desc']}")
        [curr_rows.append([f"{rct_wo['wo_num']}:{idx}",
                           f"{rct_wo['trade_grp']} {rct_wo['trade']} : {rct_wo['wo_desc']}",
                           candidate_pm_descs[idx],
                           round(random.uniform(0.35, 0.51), 2) if candidate_pm_descs[
                                                                       idx] in assigned_pm_descs else 0.0]
                          )
         for idx in tqdm(range(len(candidate_pm_descs)))
         ]

        if rct_wo_idx < 1:
            util_functions.save_dict_to_excel_workbook_with_row_formatting(
                file_path=f'{_map_file_path}_{_map_iter}.xlsx',
                headers=['mapping_code', 'rct_desc', 'pm_wo_desc', 'similarity'],
                rows=curr_rows)
        else:
            reach_limit = util_functions.append_excel_workbook(file_path=f'{_map_file_path}_{_map_iter}.xlsx',
                                                               rows=curr_rows)
            if reach_limit:
                _map_iter += 1
                util_functions.save_dict_to_excel_workbook_with_row_formatting(
                    file_path=f'{_map_file_path}_{_map_iter}.xlsx',
                    headers=['mapping_code', 'rct_desc', 'pm_wo_desc', 'similarity'],
                    rows=curr_rows)
        rct_wo_idx += 1


def main():
    # no_asset_wo_ids = [util_functions.lower_case_and_clear_white_space(wo['wo_num']) for wo in read_no_asset_work_orders()]
    # print(f'there are {len(no_asset_wo_ids)} work orders without asset')
    pm_wos = read_pm_work_orders()
    rct_wos = read_rct_work_orders()
    _rct_with_pms, _rct_without_pms = _map_rct_to_pms(rct_wos,
                                                      sorted(pm_wos, key=itemgetter('actual_comp_date'), reverse=True))
    print(
        f'\nOf all the {len(rct_wos)} RCTs, there are {len(_rct_with_pms)} RCTs with PMs against it and {len(_rct_without_pms)} RCTs without')
    return map_rct_desc_to_pm_desc(pm_wos=sorted(pm_wos, key=itemgetter('wo_desc')),
                                   rct_wos=sorted(rct_wos, key=itemgetter('wo_desc')))
    util_functions.save_dict_to_excel_workbook_with_row_formatting(file_path=SAVED_MAPPED_FILE_PATH,
                                                                   headers=MAPPED_WORK_ORDER_HEADERS,
                                                                   rows=_format_rct_pm_data_for_workbook(rct_wos,
                                                                                                         minimise=-1))


if __name__ == '__main__':
    main()
