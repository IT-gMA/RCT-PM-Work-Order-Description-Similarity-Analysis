import sys, os, math
from util_fucntions import util_functions
from operator import itemgetter
from itertools import groupby, filterfalse

PM_WORK_ORDER_EXCEL_PATH = '../../xlsx_resources/pfm_wa_report/contract_workorder_extract_pm_detailed.xlsx'
RCT_WORK_ORDER_EXCEL_PATH = '../../xlsx_resources/pfm_wa_report/rct_work_orders.xlsx'


def concatenate_subservice_to_service(sub_service: str, service: str) -> str:
    if not util_functions.are_strings_similar(sub_service, service):
        return f'{service} {sub_service}'

    trimmed_sub_service = util_functions.lower_case_and_clear_white_space(sub_service)
    trimmed_service = util_functions.lower_case_and_clear_white_space(service)
    return sub_service if trimmed_service in trimmed_sub_service else f'{service} {sub_service}'


def format_work_order_data_keys(excel_data: list) -> list:
    return [{
        'work_order_num': data['work_order_number'],
        'work_order_desc': data['description'],
        'work_order_type': data['work_type_revised'],
        'work_order_type_trimmed': util_functions.lower_case_and_clear_white_space(data['work_type_revised']),
        'work_order_status': data['work_order_status'],
        #'work_log': data['worklog'].replace("?\n", ' ').replace("\n", ' '),
        'location': data['location_full_description'],
        'trade': data['trade'],
        'trade_formatted': util_functions.clean_white_space(util_functions.remove_brackets(data['trade'])),
        'wo_count': int(data['count_of_wo']),
        'rectification_date': data['target_rectification_date'],
    } for data in excel_data]


def format_rct_work_order_data_keys(excel_data: list) -> list:
    return [{
        'work_order_num': data['work_order'],
        'work_order_desc': data['work_description'],
        'contract_code': data['contract'],
        'priority': data['prio'],
        #'work_log': data['worklog'].replace("?\n", ' ').replace("\n", ' '),
        'service_type': data['service_type'],
        'service_sub_type': data['service_sub_type'],
        'combined_service': f"{data['service_type']} {data['service_sub_type']}", #concatenate_subservice_to_service(data['service_type'], data['service_sub_type']),
        'response_sla_compliance': data['response_sla_compliance'],
        'completion_sla_compliance': data['completion_sla_compliance'],
        'createdon': data['service_request_creation_date'],
        'work_order_type': 'RCT',
        'work_order_type_trimmed': 'rct',
    } for data in excel_data]


def _get_pm_and_das_work_orders(work_orders: list, citeria=None) -> list:
    _default_citeria = ['pm', 'das']
    if citeria is None or type(citeria) != list:
        citeria = _default_citeria
    elif len(citeria) < 0:
        citeria = _default_citeria
    return [work_order for work_order in work_orders if work_order['work_order_type_trimmed'] in citeria]


def _separate_reactive_wos(work_orders: list, remove_lvm=False) -> tuple:
    non_reactive_wos = []
    reactive_wos = []
    for work_order in work_orders:
        if work_order['work_order_type_trimmed'] == 'rct':
            reactive_wos.append(work_order)
        else:
            non_reactive_wos.append(work_order)
    if remove_lvm:
        non_reactive_wos = _get_pm_and_das_work_orders(non_reactive_wos)
    return reactive_wos, non_reactive_wos


def _correlate_reactive_to_pm(reactive_wos: list, non_reactive_wos: list) -> list:
    correlated_wos = []
    for work_order in reactive_wos:
        work_order['existing_pms'] = [data for data in non_reactive_wos if data['trade'] == work_order['trade']]
        if len(work_order['existing_pms']) > 0:
            correlated_wos.append(work_order)
    return correlated_wos


def read_all_work_orders() -> tuple:
    # read from Maintenance Trend report
    work_order_json = format_work_order_data_keys(util_functions.read_excel_file(path=PM_WORK_ORDER_EXCEL_PATH, format_key=True))
    #return [print(data) for data in work_order_json], None
    reactive_wos, non_reactive_wos = _separate_reactive_wos(work_order_json, True)

    all_services = sorted(list(set(data['trade'] for data in work_order_json)))
    all_statues = sorted(list(set(data['work_order_status'] for data in work_order_json)))
    #print(len(all_services))
    #[print(service) for service in all_services]
    #correlated_wos = _correlate_reactive_to_pm(reactive_wos=reactive_wos, non_reactive_wos=non_reactive_wos)

    return reactive_wos, non_reactive_wos


def read_reactive_work_orders() -> list:
    # read from Reactive Work Order Review report
    rct_wo_json = format_rct_work_order_data_keys(util_functions.read_excel_file(path=RCT_WORK_ORDER_EXCEL_PATH, format_key=True))
    all_services = sorted(list(set([data['combined_service'] for data in rct_wo_json])))
    [print(service) for service in all_services]
    print(len(all_services))
    return rct_wo_json


def _filter_rct_work_order(trend_datas: list, rct_datas: list) -> list:
    # trend_datas refer to RCT's that come from the Maintenance Trend report
    # rct_datas refer to RCT's that come from the Reactive Work Order Review report

    filtered_rct_datas = []
    for rct_data in rct_datas:
        matched_trend_data = [trend_data for trend_data in trend_datas if rct_data['work_order_num'] == trend_data['work_order_num']]
        if len(matched_trend_data) > 0:
            matched_trend_data = matched_trend_data[0]
            rct_data['trade'] = matched_trend_data['trade']
            rct_data['wo_desc_in_trend_report'] = matched_trend_data['work_order_desc']
            rct_data['wo_count'] = matched_trend_data['wo_count']
            rct_data['location'] = matched_trend_data['location']
            rct_data['rectification_date'] = matched_trend_data['rectification_date']
            rct_data['work_order_status'] = matched_trend_data['work_order_status']
            rct_data['existing_pms'] = matched_trend_data['existing_pms']
            filtered_rct_datas.append(rct_data)

    return filtered_rct_datas


def _grp_pm_on_trade_name(pms: list) -> list:
    return util_functions.group_dict_list_by_key(pms, 'trade_formatted', 'grouped_pms')


def map_rct_service_type_to_vendor(rct_wos: list, trend_non_reactive_wos: list):
    for rct_wo in rct_wos:
        rct_wo['PMs'] = []
        rct_wo['num_pms'] = 0
        matched_pms = [pm for pm in trend_non_reactive_wos if util_functions.are_strings_similar(pm['trade_formatted'], rct_wo['service_sub_type'])]
        if len(matched_pms) < 1:
            continue
        for matched_pm in matched_pms:
            for _pm in matched_pm['grouped_pms']:

                rct_wo['PMs'].append(_pm)
        rct_wo['num_pms'] = len(rct_wo['PMs'])


def _format_rct_pm_data_for_workbook(rct_wos: list) -> list:
    new_data = []
    for rct in sorted(rct_wos, key=itemgetter('num_pms')):
        if rct['num_pms'] < 1:
            new_data.append([rct['work_order_num'], rct['work_order_desc'], rct['contract_code'],
                             rct['priority'], rct['service_type'], rct['service_sub_type'],
                             rct['response_sla_compliance'],
                             rct['completion_sla_compliance'], rct['createdon'], rct['num_pms'],
                             None, None, None, None, None, None])
            continue
        for idx in range(0, 2):
            try:
                _pm = rct['PMs'][idx]
            except IndexError:
                continue

            if idx == 0:
                new_data.append([rct['work_order_num'], rct['work_order_desc'], rct['contract_code'],
                                 rct['priority'], rct['service_type'], rct['service_sub_type'], rct['response_sla_compliance'],
                                 rct['completion_sla_compliance'], rct['createdon'], rct['num_pms'],
                                 _pm['work_order_num'], _pm['work_order_desc'], _pm['location'], _pm['work_order_status'], _pm['trade'], _pm['rectification_date']])
            else:
                new_data.append([None, None, None, None, None, None, None, None, None, None,
                                 _pm['work_order_num'], _pm['work_order_desc'], _pm['location'],
                                 _pm['work_order_status'], _pm['trade'], _pm['rectification_date']])
    return new_data


def main():
    trend_reactive_wos, trend_non_reactive_wos = read_all_work_orders()
    rct_wos = read_reactive_work_orders()
    map_rct_service_type_to_vendor(rct_wos=rct_wos, trend_non_reactive_wos=_grp_pm_on_trade_name(trend_non_reactive_wos))
    rct_with_pms = [rct_wo for rct_wo in rct_wos if rct_wo['num_pms'] > 0]
    rct_without_pms = [rct_wo for rct_wo in rct_wos if rct_wo['num_pms'] < 1]
    '''for rct_wo in rct_with_pms:
        print(rct_wo)'''

    print(f"Out of of {len(rct_wos)} RCTs, there are {len(rct_with_pms)} RCTs with PMs and {len(rct_without_pms)} RCTs without PMs")
    _formatted_rct_datas = _format_rct_pm_data_for_workbook(rct_wos=rct_wos)
    _headers = ['RCT Work Order Number', 'RCT Description', 'Contract Code', 'Priority', 'Service Type', 'Service Sub Type', 'Response SLA Compliance', 'Completion SLA Compliance', 'Date of Issue', 'Number of Matched PMs', 'PM Work Order Number', 'PM Work Order Description', 'Location', 'Status', 'Trade', 'Date of Rectification']
    '''_prev_idx = 0
    i = 0
    for idx in range(0, len(_formatted_rct_datas), int(len(_formatted_rct_datas) / 15)):
        if idx == 0:
            continue
        print(f'_formatted_rct_datas[{_prev_idx}:{idx}]')
        util_functions.save_dict_to_excel_workbook_with_row_formatting(
            file_path=f'../saved_resources/rct_with_pms_{i}.xlsx',
            headers=_headers,
            rows=_formatted_rct_datas[_prev_idx:idx])

        _prev_idx = idx
        i += 1'''
    util_functions.save_dict_to_excel_workbook_with_row_formatting(
        file_path='../saved_resources/rct_with_pms.xlsx',
        headers=_headers,
        rows=_formatted_rct_datas)


if __name__ == '__main__':
    main()
