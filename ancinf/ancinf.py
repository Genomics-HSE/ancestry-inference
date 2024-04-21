import click
import numpy as np
import time

from .utils import simulate as sim 
from .utils import runheuristic


#@click.option("--count", default=1, help="Number of greetings.")
#@click.option("--name", prompt="Your name", help="The person to greet.")
#click.echo(f"Hello, {folder}, {out}!")



@click.group()
def cli():
    pass


#STAGE1 GETPARAMS
@cli.command()
@click.argument("datadir")
@click.argument("workdir")
@click.option("--infile", default="project.ancinf", help="Project file, defaults to project.ancinf")
@click.option("--outfile", default=None, help="Output file with simulation parameters, defaults to project file with '.params' extension")
def getparams(datadir, workdir, infile, outfile):
    """Collect parameters of csv files in the DATADIR listed in project file from WORKDIR"""    
    if outfile is None:
        #try to remove .ancinf from infile
        position = infile.find('.ancinf')
        if position>0:
            outfile = infile[:position]+'.params'
        else:
            outfile = infile+'.params'
    sim.collectandsaveparams(datadir, workdir, infile, outfile)
    print("Finished!")

    
#STAGE1' PREPROCESS    
@cli.command()
@click.argument("datadir")
@click.argument("workdir")
@click.option("--infile", default="project.ancinf", help="Project file, defaults to project.ancinf")
@click.option("--outfile", default=None, help="Output file with experiment list, defaults to project file with '.explist' extension")
@click.option("--seed", default=2023, help="Random seed")
def preprocess(datadir, workdir, infile, outfile, seed):
    """Filter datsets from DATADIR, generate train-val-test splits and experiment list file in WORKDIR"""    
    if outfile is None:
        #try to remove .ancinf from infile
        position = infile.find('.ancinf')
        if position>0:
            outfile = infile[:position]+'.explist'
        else:
            outfile = infile+'.explist'
    rng = np.random.default_rng(seed)
    start = time.time()
    sim.preprocess(datadir, workdir, infile, outfile, rng)
    print(f"Finished! Total {time.time()-start:.2f}s")


#STAGE 2 SIMULATE
@cli.command()
@click.argument("workdir")
@click.option("--infile", default="project.params", help="File with simulation parameters, defaults to project.params")
@click.option("--outfile", default=None, help="Output file with experiment list, defaults to project file with '.explist' extension")
@click.option("--seed", default=2023, help="Random seed")
def simulate(workdir, infile, outfile, seed):
    """Generate ibd graphs, corresponding slpits and experiment list file for parameters in INFILE"""    
    if outfile is None:
        #try to remove .ancinf from infile
        position = infile.find('.params')
        if position>0:
            outfile = infile[:position]+'.explist'
        else:
            outfile = infile+'.explist'
            
    rng = np.random.default_rng(seed)
    start = time.time()
    sim.simulateandsave(workdir, infile, outfile, rng)
    print(f"Finished! Total {time.time()-start:.2f}s")
    
    
#STAGE3 HEURISTICS
@cli.command()
@click.argument("workdir")
@click.option("--infile", default="project.explist", help="File with experiment list, defaults to project.explist")
@click.option("--outfile", default=None, help="File with classification metrics, defaults to project file with '.result' extension")
@click.option("--seed", default=2023, help="Random seed")
def heuristics(workdir, infile, outfile, seed):
    """Run heuristics"""
    rng = np.random.default_rng(seed)
    if outfile is None:
        #try to remove .ancinf from infile
        position = infile.find('.explist')
        if position>0:
            outfile = infile[:position]+'.result'
        else:
            outfile = infile+'.result'
    start = time.time()
    runheuristic.runandsaveheuristics(workdir, infile, outfile, rng)
    print(f"Finished! Total {time.time()-start:.2f}s")
    
#STAGE4 GNN    
@cli.command()
@click.argument("workdir")
@click.option("--infile", default="project.explist", help="File with experiment list, defaults to project.explist")
@click.option("--outfile", default=None, help="File with classification metrics, defaults to project file with '.result' extension")
@click.option("--seed", default=2023, help="Random seed")
def gnn(workdir, infile, outfile, seed):
    """Run gnns"""     
    rng = np.random.default_rng(seed)  
    if outfile is None:
        #try to remove .ancinf from infile
        position = infile.find('.explist')
        if position>0:
            outfile = infile[:position]+'.result'
        else:
            outfile = infile+'.result'
    start = time.time()
    sim.runandsavegnn(workdir, infile, outfile, rng)
    print(f"Finished! Total {time.time()-start:.2f}s.")

    
#STAGE5 HEURISTICS AND GNNS
@cli.command()
@click.argument("workdir")
@click.option("--infile", default="project.explist", help="File with experiment list, defaults to project.explist")
@click.option("--outfile", default=None, help="File with classification metrics, defaults to project file with '.result' extension")
@click.option("--seed", default=2023, help="Random seed")
def runall(workdir, infile, outfile, seed):
    """Run heuristics, gnns and fc networks"""     
    rng = np.random.default_rng(seed)  
    if outfile is None:
        #try to remove .ancinf from infile
        position = infile.find('.explist')
        if position>0:
            outfile = infile[:position]+'.result'
        else:
            outfile = infile+'.result'
    start = time.time()            
    sim.runandsaveall(workdir, infile, outfile, rng)
    print(f"Finished! Total {time.time()-start:.2f}s.")    

#INFERENCE STAGES
    
    
#GNN DEBUG&TEST STAGES
@cli.command()
@click.argument("workdir")
@click.option("--infile", default="project.explist", help="File with experiment list, defaults to project.explist")
@click.option("--outfile", default=None, help="File with classification metrics, defaults to project file with '.result' extension")
@click.option("--seed", default=2023, help="Random seed")
def gnncleancheck(workdir, traindfname, infdfname):
    '''
    #-> traindataframe, one-node-infdataframe, node_idname, model, weights 
    #for every node in inferencedf
    #1. create train+one-node-infdataframe
    #2. send to inference
    #3. compare with true 
    '''
    pass
    
    
def preparecleancheck(workdir, datafilename, cleanshare, seed):
    '''
        get dataframe and split into train-test-val (1-cleanshare for every class) 
        and clean-test (share for every class)
    '''
    rng = np.random.default_rng(seed)  
    position = datafilename.find('.csv')
    if position>0:
        outfile = infile[:position]+'.result'
    else:
        outfile = infile+'.result'
    tranfilename = 
    cleanfilename = 
    sim.preparecleancheck(workdir, datafilename, trainfilename, cleanfilename, cleanshare, rng)
    
def main():
    cli()