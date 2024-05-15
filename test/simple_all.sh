python3 -m ancinf preprocess ./datasets/ ./workdir/test1/ --infile arrakis_all.ancinf
python3 -m ancinf crossval ./workdir/test1/ --infile arrakis_all.explist --gpu 1