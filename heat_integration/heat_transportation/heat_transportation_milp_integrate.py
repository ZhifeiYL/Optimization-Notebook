import os

import pandas as pd

from util.util import get_root_dir

__all__ = ['generate_gams_data', 'build_gams_file']


def generate_gams_data(data):
    # Read the stream information
    streams = pd.read_csv(data['streams'])
    hot_streams = streams[(streams['Type'] == 'Hot')]['ID'].tolist()
    cold_streams = streams[(streams['Type'] == 'Cold')]['ID'].tolist()

    hot_utilities = streams[(streams['Type'] == 'Hot') & (streams['Utility'] == 1)]['ID'].tolist()
    cold_utilities = streams[(streams['Type'] == 'Cold') & (streams['Utility'] == 1)]['ID'].tolist()

    hot_non_utilities = [stream for stream in hot_streams if stream not in hot_utilities]
    cold_non_utilities = [stream for stream in cold_streams if stream not in cold_utilities]

    intervals = [str(i) for i in range(1, len(pd.read_csv(data['tabH'])) + 1)]

    # Read valid combinations
    valid_ht = pd.read_csv(data['utility_interval']).query('Utility in @hot_utilities')
    valid_ct = pd.read_csv(data['utility_interval']).query('Utility in @cold_utilities')

    # Read tables
    tabHs = pd.read_csv(data['tabH']).set_index('i')
    tabCs = pd.read_csv(data['tabC']).set_index('i')

    # Read costs
    costs = pd.read_csv(data['utility_cost']).set_index('Utility')

    # Generate the GAMS data string
    data_string = ""

    # Adding hot and cold streams
    data_string += f"Sets\nh /{', '.join(hot_streams)}/\n"
    data_string += f"c /{', '.join(cold_streams)}/\n\n"

    # Adding hot streams and cold streams that are not utilities
    data_string += f"hs(h) /{', '.join(hot_non_utilities)}/\n"
    data_string += f"cs(c) /{', '.join(cold_non_utilities)}/\n\n"

    # Adding utilities
    data_string += f"ht(h) /{', '.join(hot_utilities)}/\n"
    data_string += f"ct(c) /{', '.join(cold_utilities)}/\n\n"

    # Adding intervals
    data_string += f"i /{', '.join(intervals)}/;\n\n"

    data_string += f"alias (i, ip);\n\n"

    # Adding valid combinations
    data_string += "Set valid_combinations_ht(ht, i) /"
    for index, row in valid_ht.iterrows():
        data_string += f"{row['Utility']}.{row['Interval']}"
        if index != valid_ht.index[-1]:
            data_string += ", "
    data_string += "/;\n\n"

    data_string += "Set valid_combinations_ct(ct, i) /"
    for index, row in valid_ct.iterrows():
        data_string += f"{row['Utility']}.{row['Interval']}"
        if index != valid_ct.index[-1]:
            data_string += ", "
    data_string += "/;\n\n"

    # Adding tabHs as parameters
    data_string += "Parameter tabHs(i,h) Hot streams values with utility default to 0 /\n"
    for idx, row in tabHs.iterrows():
        for col, value in row.items():
            data_string += f"{idx}.{col} {value}\n"
    data_string += "/;\n\n"

    # Adding tabCs as parameters
    data_string += "Parameter tabCs(i,c) Cold streams values with utility default to 0 /\n"
    for idx, row in tabCs.iterrows():
        for col, value in row.items():
            data_string += f"{idx}.{col} {value}\n"
    data_string += "/;\n\n"

    # Adding costs
    # Declaring cost parameters
    data_string += "Parameter\n"
    data_string += "Qh(i,h) Hot stream duty \n"
    data_string += "Qc(i,c) Cold stream duty \n"
    data_string += "Cm(ht) Utility costs for hot utilities\n"
    data_string += "Cn(ct) Utility costs for cold utilities;\n\n"

    # Assigning values to cost parameters for hot utilities
    for utility, cost in costs.query('Utility in @hot_utilities').iterrows():
        data_string += f"Cm('{utility}') = {cost['Cost']};\n"

    # Assigning values to cost parameters for cold utilities
    for utility, cost in costs.query('Utility in @cold_utilities').iterrows():
        data_string += f"Cn('{utility}') = {cost['Cost']};\n"

    data_string += "\n"

    return data_string


def build_gams_file(case_name, data, case_dir, body):
    directory = get_root_dir() + case_dir
    model_data = generate_gams_data(data)

    # Define the static parts of the GAMS model
    with open(directory + body, 'r') as body_file:
        model_body = body_file.read()

    export_statement = "\nexecute_unload \"{}.gdx\";".format(case_name)
    full_model = model_data + model_body + export_statement

    # Write the combined model to a GAMS file
    directory = get_root_dir() + case_dir
    with open(directory + case_name + '.gms', 'w') as file:
        file.write(full_model)

    return
