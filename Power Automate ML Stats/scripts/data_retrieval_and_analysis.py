from util_fucntions import util_functions
from tqdm import tqdm
import itertools
from itertools import groupby
from operator import itemgetter

METRIC_REPORT_PATH = '../JSON metric report/Metrics report - Harmonised PM Desc Classification S v1p2.json'
SAVED_FILE_PATH = '../Workbook metric report/Metrics report - Harmonised PM Desc Classification S v1p2.xlsx'
HEADERS = ['Class', 'Accuracy', 'Baseline Accuracy', 'Precision', 'Recall', 'F1 Score', 'F1 Score (Baseline)',
           'Num False Positives',
           'Num False negatives']
#HEADERS = ['Entity', 'Precision', 'Recall', 'F1']


def _find_metric(metric_name: str, metric_list: list, to_percentage=False, round_to=2):
    score = [metric for metric in metric_list if metric['name'] == metric_name][0]['value']
    return score if type(score) == int else round(score * 100 if to_percentage else score, round_to)


def _format_to_dictionaries(metric_report_json: list) -> list:
    return [
        {
            'class': data['id'],
            'accuracy': _find_metric('accuracy', data['metrics'], True),
            'base_accuracy': _find_metric('acuracyBaseline', data['metrics'], True),
            'precision': _find_metric('precision', data['metrics'], True),
            'recall': _find_metric('recall', data['metrics'], True),
            'f1': _find_metric('f1Score', data['metrics'], True),
            'base_f1': _find_metric('f1ScoreBaseline', data['metrics'], True),
            'num_false_pos': _find_metric('numberOfFalsePositives', data['metrics']),
            'num_false_negv': _find_metric('numberOfFalseNegatives', data['metrics']),
        }
        for data in sorted(metric_report_json, key=itemgetter('id'))
    ]


def _analyse_data(stat_data: list) -> None:
    lowest_precision_class = sorted(stat_data, key=itemgetter('precision'))[0]
    best_precision_class = sorted(stat_data, key=itemgetter('precision'), reverse=True)[0]
    lowest_recall_class = sorted(stat_data, key=itemgetter('recall'))[0]
    best_recall_class = sorted(stat_data, key=itemgetter('recall'), reverse=True)[0]
    lowest_f1_class = sorted(stat_data, key=itemgetter('f1'))[0]
    best_f1_class = sorted(stat_data, key=itemgetter('f1'), reverse=True)[0]

    worst_false_postive_class = sorted(stat_data, key=itemgetter('num_false_pos'), reverse=True)[0]
    best_false_postive_class = sorted(stat_data, key=itemgetter('num_false_pos'))[0]

    worst_false_negv_class = sorted(stat_data, key=itemgetter('num_false_negv'), reverse=True)[0]
    best_false_negv_class = sorted(stat_data, key=itemgetter('num_false_negv'))[0]

    for data in stat_data:
        data['accuracy_diff'] = data['accuracy'] - data['base_accuracy']
        data['false_pos_nevg_diff'] = abs(data['num_false_pos'] - data['num_false_negv'])
    lowest_accuracy_class = sorted(stat_data, key=itemgetter('accuracy_diff'))[0]
    best_accuracy_class = sorted(stat_data, key=itemgetter('accuracy_diff'), reverse=True)[0]
    highest_false_pos_nevg_diff_class = sorted(stat_data, key=itemgetter('false_pos_nevg_diff'), reverse=True)[0]
    lowest_false_pos_nevg_diff_class = sorted(stat_data, key=itemgetter('false_pos_nevg_diff'))[0]

    print(
        f"_________________\n"
        f"Class with the worst Difference between Accuracy and Baseline Accuracy is {lowest_accuracy_class['class']} at {abs(lowest_accuracy_class['accuracy_diff'])}\n"
        f"Class with the worst Precision is {lowest_precision_class['class']} at {lowest_precision_class['precision']}\n"
        f"Class with the worst Recall is {lowest_recall_class['class']} at {lowest_recall_class['recall']}\n"
        f"Class with the lowest F1 score is {lowest_f1_class['class']} at {lowest_f1_class['f1']}\n"
        f"Class with the highest number of False Positives is {worst_false_postive_class['class']} at {worst_false_postive_class['num_false_pos']}\n"
        f"Class with the highest number of False Negatives is {worst_false_negv_class['class']} at {worst_false_negv_class['num_false_negv']}\n"
        f"Class with the largest difference between False Negatives and Positives is {highest_false_pos_nevg_diff_class['class']} at {highest_false_pos_nevg_diff_class['false_pos_nevg_diff']}: {highest_false_pos_nevg_diff_class['num_false_negv']} - {highest_false_pos_nevg_diff_class['num_false_pos']} False Negative - Positive instances\n"
        f"_________________________________________________________________________\n"
        f"Class with the best Difference between Accuracy and Baseline Accuracy is {best_accuracy_class['class']} at {abs(best_accuracy_class['accuracy_diff'])}\n"
        f"Class with the best Precision is {best_precision_class['class']} at {best_precision_class['precision']}\n"
        f"Class with the best Recall is {best_recall_class['class']} at {best_recall_class['recall']}\n"
        f"Class with the best F1 score is {best_f1_class['class']} at {best_f1_class['f1']}\n"
        f"Class with the lowest number of False Positives is {best_false_postive_class['class']} at {best_false_postive_class['num_false_pos']}\n"
        f"Class with the lowest number of False Negatives is {best_false_negv_class['class']} at {best_false_negv_class['num_false_negv']}\n"
        f"Class with the smallest difference between False Negatives and Positives is {lowest_false_pos_nevg_diff_class['class']} at {lowest_false_pos_nevg_diff_class['false_pos_nevg_diff']}"
        f": {lowest_false_pos_nevg_diff_class['num_false_negv']} - {lowest_false_pos_nevg_diff_class['num_false_pos']} False Negative - Positive instances\n"
    )


def save_data_to_excel(formatted_data: list) -> None:
    util_functions.save_dict_to_excel_workbook_with_row_formatting(file_path=SAVED_FILE_PATH,
                                                                   headers=HEADERS,
                                                                   rows=[
                                                                       [
                                                                           data['class'], data['accuracy'],
                                                                           data['base_accuracy'],
                                                                           #data['entity'],
                                                                           data['precision'], data['recall'],
                                                                           data['f1'], data['base_f1'],
                                                                           data['num_false_pos'],
                                                                           data['num_false_negv']
                                                                       ]
                                                                       for data in formatted_data])


def main():
    metric_report_json = util_functions.read_from_json_file(file_path=METRIC_REPORT_PATH)
    for data in metric_report_json['details']:
        print(data)
    formatted_data = _format_to_dictionaries(metric_report_json=metric_report_json['details'])
    _analyse_data(formatted_data)
    #save_data_to_excel(formatted_data)


if __name__ == '__main__':
    main()
