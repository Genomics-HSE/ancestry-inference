python3 -m ancinf preprocess ./datasets/ ./workdir/test1/ --infile arrakis_baseline.ancinf
python3 -m ancinf crossval ./workdir/test1/ --infile arrakis_baseline.explist --gpu 1