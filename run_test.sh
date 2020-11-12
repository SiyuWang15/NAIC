time=$(date "+%m%d-%H-%M-%S")

python main.py --runner testing --log_prefix $time --Pn 32 & 
python main.py --runner testing --log_prefix $time --Pn 8