time=$(date "+%m%d-%H-%M-%S")

CUDA_VISIBLE_DEVICES=7 python main.py --runner validation --log_prefix $time --Pn 32 
