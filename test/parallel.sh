python3 -m ancinf preprocess ./datasets/ ./workdir/test5/ --infile simple.ancinf
python3 -m ancinf crossval ./workdir/test5 --infile simple.explist --processes 4 --gpucount 4 --fromsplit 0 --tosplit 10
