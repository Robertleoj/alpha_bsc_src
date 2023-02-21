import os
from sys import argv

# train_cmd

cwd = os.getcwd()

def self_play(gen:int, dep:str=None) ->str:

    cmd = f"cd {cwd} && sbatch --job-name=self_play{gen} --output=self_play{gen}.log" 
    if dep is not None:
        cmd += f" --dependency=aftercorr:{dep}"
    cmd += f" self_play.sh"

    job_id = os.popen(cmd).read()
    print(f"started self_play job {job_id}")
    return job_id

def train(gen:int, dep:str=None) ->str:

    cmd = f"{cwd} && sbatch --job-name=train{gen} --output=train{gen}.log" 
    if dep is not None:
        cmd += f" --dependency=aftercorr:{dep}"
    cmd += f" train.sh"

    job_id = os.popen(cmd).read()
    print(f"started train job {job_id}")
    return job_id

def single_cycle(gen, dep:str=None) -> str:
    dep = self_play(gen, dep)
    dep = train(gen, dep)
    return dep

def cycles(gen, num_gens, dep):
    for g in range(gen, gen+num_gens):
        dep = single_cycle(g, dep)
    

def main():
    first_gen = 0

    if len(argv) >= 2:
        first_gen = int(argv[1])

    num_generations = 5

    if len(argv) >= 3:
        num_generations = int(argv[2])

    first_dep = None
    if len(argv) >= 4:
        first_dep = argv[3]

    cycles(first_gen, num_generations, first_dep)

main()






