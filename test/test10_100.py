import json
import os
import subprocess
# from .console.run_console import run_console_command


def run_console_command(command):
    process = subprocess.Popen(command.split(' '), stdout=subprocess.PIPE, text=True, bufsize=1, start_new_session=True)
    lst_lines = []
    for line in process.stdout:
        print(line, end='')
        lst_lines.append(line)
    return lst_lines    
    process.wait()

    
test_name = 'arrakis100-10'

command1 = f"python3 -m ancinf simulate ./workdir/test4/ --infile {test_name}.params"   
command2 = f"python3 -m ancinf getparams ./datasets/ ./workdir/test4/ --infile arrakis100-10_Arrakis_exp0.ancinf"
command3 = f"python3 -m ancinf getparams ./datasets/ ./workdir/test4/ --infile arrakis100-10_Arrakis_exp0.ancinf --outfile arrakis100-10_Arrakis_exp02.params"
command4 = f"diff ./workdir/test4/arrakis100-10_Arrakis_exp0.params.reference ./workdir/test4/arrakis100-10_Arrakis_exp0.params"
run_console_command(command1)
run_console_command(command2)
lst = run_console_command(command4)
if lst == []:
    print("Params files are equal")
else:
    print("Params files are not equal")