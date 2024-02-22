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
@click.option("--out", default="paramfile.json", help="Output file.")
def getparams(folder, out):
    """Collect parameters of all csv files in the FOLDER"""    
    sim.collectandsaveparams(folder, out)

    
@cli.command()
@click.argument("paramfile")
@click.option("--folder", default=".", help="Output folder.")
def simulate(paramfile, folder):
    """Generate ibd graphs for parameters in PARAMFILE"""
    sim.simulateandsave(paramfile, folder)

    

def main():
    cli()