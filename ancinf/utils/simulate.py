import json
import os

def collectparams(folder):
    #get every csv in the folder and compute its parameters
    #maybe some other parameters
    dummy = {"CR.csv": {"pop_names":["pop1","pop2","pop3"],
                        "pop_sizes":[1000,1000,1000],
                        "edge_probability":[[1,0,0],[0,1,0],[0,0,1]]
                       },
             "NC.csv": {"pop_names":["pop1","pop2","pop3"],
                        "pop_sizes":[1000,1000,1000],
                        "edge_probability":[[1,0,0],[0,1,0],[0,0,1]]
                       }
            }
    return dummy

    
def collectandsaveparams(folder, outfile):
    print(f"Collecting parameters for datasets from {folder}")
    collected = collectparams(folder)    
    with open(outfile,"w") as f:
        json.dump(collected,f)

    
def simulate(params):
    df = ""
    return df
    
    

def simulateandsave(paramfile, outfolder):
    print(f"Running simulations for parameters from {paramfile}")
    #create separate csv file for every record in paramfile 
    with open(paramfile,'r') as f:
        dct = json.load(f)
    for csvname in dct:
        simdf = simulate(dct[csvname])
        fname = os.path.join(outfolder, 'sim'+csvname)
        with open(fname, 'w') as f:
            #save dataframe here
            pass
        
    

if __name__=="__main":
    print("just a test")