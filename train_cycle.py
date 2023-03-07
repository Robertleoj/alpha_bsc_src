import os
from sys import argv

# train_cmd

cwd = "/home/gimli/AlphaBSc/alpha_bsc_src"

def self_play(gen:int, dep:str=None) ->str:
    print("starting self_play job")

    cmd = f"cd {cwd} && sbatch --job-name=self_play{gen} --output=./logs/self_play{gen}.log" 
    if dep is not None:
        cmd += f" --dependency=aftercorr:{dep}"
    cmd += f" self_play.sh"
    print("running command")
    print(cmd)

    outp = os.popen(cmd).read()
    print("output: ")
    print(outp)
    
    job_id = outp.strip().split(' ')[-1]

    print(f"started self_play job {job_id}")
    return job_id

def evaluate(gen:int, dep:str=None) ->str:
    print("starting evaluation job")

    cmd = f"cd {cwd} && sbatch --job-name=evaluate{gen} --output=./logs/eval{gen}.log" 
    if dep is not None:
        cmd += f" --dependency=aftercorr:{dep}"
    cmd += f" eval.sh"
    print("running command")
    print(cmd)

    outp = os.popen(cmd).read()
    print("output: ")
    print(outp)
    
    job_id = outp.strip().split(' ')[-1]

    print(f"started eval job {job_id}")
    return job_id



def train(gen:int, dep:str=None) ->str:
    print('starting train job')

    cmd = f"cd {cwd} && sbatch --job-name=train{gen} --output=./logs/train{gen}.log" 
    if dep is not None:
        cmd += f" --dependency=aftercorr:{dep}"
    cmd += f" train.sh"

    print('running command')
    print(cmd)
    outp = os.popen(cmd).read()
    print("output: ")
    print(outp)
    
    job_id = outp.strip().split(' ')[-1]

    print(f"started train job {job_id}")
    return job_id

def single_cycle(gen, dep:str=None) -> str:
    dep = evaluate(gen, dep)
    dep = self_play(gen, dep)
    dep = train(gen, dep)
    return dep

def cycles(gen, num_gens, dep):
    for g in range(gen, gen+num_gens):
        dep = single_cycle(g, dep)
    evaluate(gen, dep)
    

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






