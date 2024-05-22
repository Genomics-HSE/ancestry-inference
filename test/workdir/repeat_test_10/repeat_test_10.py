import json
import os
import subprocess
# from .console.run_console import run_console_command

def repeat_test_10(test_name):
    command1 = f"python3 -m ancinf simulate ./workdir/repeat_test_10 --infile {test_name}.params"
    repeat_n = 2

    def run_console_command(command):
        process = subprocess.Popen(command.split(' '), stdout=subprocess.PIPE, text=True, bufsize=1, start_new_session=True)

        for line in process.stdout:
            print(line, end='')

        process.wait()

    def check_people_entry(arrakis_data, arrakis_init_data):
        pop_names = arrakis_init_data['datasets']['Arrakis']['pop_names']
        pop_sizes = arrakis_init_data['datasets']['Arrakis']['pop_sizes']
        split_count = arrakis_init_data['crossvalidation']['split_count']

        lst_pop = []

        for i in range(split_count):
            temp = []
            for j in range(len(pop_names)):
                temp.append([])
            lst_pop.append(temp)

        for i in range(split_count):
            for j in range(len(pop_names)):
                lst_pop[i][j] += arrakis_data['partitions'][i]['test'][pop_names[j]]
                lst_pop[i][j] += arrakis_data['partitions'][i]['train'][pop_names[j]]
                lst_pop[i][j] += arrakis_data['partitions'][i]['val'][pop_names[j]]            
        return(lst_pop)

    def run_n_times(n):
        lst_repeat = []
        for i in range(n):
            run_console_command(command1)
            parent_dir = os.path.dirname(os.path.abspath(__file__))

            with open(os.path.join(parent_dir, f"{test_name}_Arrakis_exp0.split"), "r") as f:
                arrakis_data1 = json.load(f)

            with open(os.path.join(parent_dir, f"{test_name}.params"), "r") as f1:
                arrakis_init_data1 = json.load(f1)

            lst_repeat.append(check_people_entry(arrakis_data1, arrakis_init_data1))
        return(lst_repeat)

    lst_repeat = run_n_times(repeat_n)
    count = 0
    for i in range(len(lst_repeat)):
        if lst_repeat[i] == lst_repeat[0]:
                   count += 1
    if count == repeat_n:
        print('Массивы испытуемых совпали')
        return(0)
    else:
        print('Массивы испытуемых не совпали')
        for i in range(len(lst_repeat)):
            print(lst_repeat[i])
        return(1)
            