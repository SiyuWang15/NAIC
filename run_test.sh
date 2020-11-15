time=$(date "+%m%d-%H-%M-%S")

# python main.py --runner testing --log_prefix $time --Pn 32 & 
CUDA_VISIBLE_DEVICES=2 python main.py --runner testing --log_prefix $time --Pn 8