import os
import pandas as pd


def kover_report_parser(inpt_path):
    files_list = [i for i in os.walk(inpt_path)][0][2]
    files_list = [i for i in files_list if i[0] != '.']
    output_list = []
    report_list = ['Sensitivity', 'Specificity', 'True Positives', 'True Negatives', 'False Positives',
                   'False Negatives']
    temp_dict = {}
    SNP_list = []
    for i in files_list:
        print(i)
        flist = open(os.path.join(inpt_path, i)).readlines()
        parsing = False
        for line in flist:
            if line.startswith('max_rules'):
                temp_dict['Max_rule_size'] = line.split(':')[1].lstrip().split('\n')[0]
            if line.startswith('Maximum number of rules'):
                temp_dict['Rule_size'] = line.split(':')[1].lstrip().split('\n')[0]
            if line.startswith('Presence'):
                SNP_list.append(line.split('(')[1].split(')')[0].strip('_'))
                temp_dict['SNP'] = SNP_list
            if parsing:
                temp_dict['Name'] = 'Kover'
                temp_dict['Drug'] = i.split('_')[0].split('.txt')[0]
                if line.split(':')[0] in report_list:
                    temp_dict[line.split(':')[0]] = float(line.split(':')[1].lstrip())
            if line.startswith("Metrics (testing data)"):
                parsing = True
        pd.DataFrame([temp_dict]).to_csv(os.path.join(inpt_path, "{}_kover_results.csv".format(temp_dict['Drug'])),
                                         index=False, encoding='utf-8-sig')
        output_list.append(temp_dict)
        SNP_list = []
        temp_dict = {}
    pd.DataFrame(output_list).to_csv(os.path.join(inpt_path, "kover.csv"), index=False, encoding='utf-8-sig')


if __name__ == '__main__':
    input_path = os.path.join(os.getcwd(), 'kover_report')
    print(input_path)
    kover_report_parser(input_path)
