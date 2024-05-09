python3 -m ancinf preprocess ./datasets/ ./workdir/test1/ --infile arrakis_no_gnn.ancinf
python3 -m ancinf crossval ./workdir/test1/ --infile arrakis_no_gnn.explist --gpu 1