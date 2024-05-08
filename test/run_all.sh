echo Hello!
echo 1. Check preprocessing
python3 -m ancinf preprocess ./datasets/ ./workdir/test1/ --infile arrakis.ancinf
echo 2. Check getting parameters
python3 -m ancinf getparams ./datasets/ ./workdir/test2/ --infile arrakis2p.ancinf
python3 -m ancinf getparams ./datasets/ ./workdir/test2/ --infile arrakis3p.ancinf
echo 3. Check simulations
python3 -m ancinf simulate ./workdir/test3/ --infile arrakis.params