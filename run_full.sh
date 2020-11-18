time=$(date "+%m%d-%H-%M-%S")

# CUDA_VISIBLE_DEVICES=6 python main.py --runner validation --log_prefix $time --Pn 32 &
CUDA_VISIBLE_DEVICES=4 python main.py --runner testing --time $time