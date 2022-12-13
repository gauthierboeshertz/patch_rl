
#bsub -n 5 -W 4:00 -R "rusage[mem=40000,ngpus_excl_p=1]"  python3 -m scripts.train_dynamics device=cuda recons_loss=False  

#sleep 120

bsub -n 5 -W 4:00 -R "rusage[mem=40000,ngpus_excl_p=1]"  python3 -m scripts.train_dynamics device=cuda recons_loss=False dynamics.regularizer_weight=0 dynamics.num_attention_layers=1

sleep 120

bsub -n 5 -W 4:00 -R "rusage[mem=40000,ngpus_excl_p=1]"  python3 -m scripts.train_dynamics device=cuda recons_loss=False  dynamics.num_attention_layers=1

#sleep 120

#bsub -n 5 -W 4:00 -R "rusage[mem=40000,ngpus_excl_p=1]"  python3 -m scripts.train_dynamics device=cuda recons_loss=False dynamics.regularizer_weight=0 dynamics.num_attention_layers=1 dynamics.num_heads=2 

#sleep 120

#bsub -n 5 -W 4:00 -R "rusage[mem=40000,ngpus_excl_p=1]"  python3 -m scripts.train_dynamics device=cuda recons_loss=False dynamics.num_attention_layers=1 dynamics.num_heads=2

#sleep 120


#bsub -n 5 -W 4:00 -R "rusage[mem=40000,ngpus_excl_p=1]"  python3 -m scripts.train_dynamics device=cuda recons_loss=False dynamics.regularizer_weight=0 dynamics.num_attention_layers=1 dynamics.num_heads=2  dynamics.temperature=0.2

#sleep 120

#bsub -n 5 -W 4:00 -R "rusage[mem=40000,ngpus_excl_p=1]"  python3 -m scripts.train_dynamics device=cuda recons_loss=False dynamics.num_attention_layers=1 dynamics.num_heads=2 dynamics.temperature=0.2

#sleep 120
