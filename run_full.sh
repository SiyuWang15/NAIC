time=$(date "+%m%d-%H-%M-%S")

# CUDA_VISIBLE_DEVICES=6 python main.py --runner validation --log_prefix $time --Pn 32 &
CUDA_VISIBLE_DEVICES=2 python main.py --runner testing --Pn 32 --time $time