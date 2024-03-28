import click

from .utils import simulate as sim 
from .utils import runheuristic


#@click.option("--count", default=1, help="Number of greetings.")
#@click.option("--name", prompt="Your name", help="The person to greet.")
#click.echo(f"Hello, {folder}, {out}!")



@click.group()
def cli():
    pass


@cli.command()
@click.argument("folder")
@click.option("--override_popsizes", default=None, help="Output file.")
@click.option("--out", default="paramfile.json", help="Output file.")
def getparams(folder, out, override_popsizes):
    """Collect parameters of all csv files in the FOLDER"""    
    sim.collectandsaveparams(folder, out, override_popsizes)

    
@cli.command()
@click.argument("paramfile")
@click.option("--folder", default=".", help="Output folder.")
@click.option("--seed", default=2023, help="Random seed.")
def simulate(paramfile, folder, seed):
    """Generate ibd graphs for parameters in PARAMFILE"""    
    sim.simulateandsave(paramfile, folder, seed)

    

def main():
    cli()