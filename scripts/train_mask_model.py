import argparse
from cProfile import label
from stable_baselines3.common.utils import set_random_seed
from tqdm import tqdm
import numpy as np
from dynamics_model import DynamicsModel
from torch.utils.data import DataLoader, TensorDataset
import torch
from progress.bar import Bar
import logging
import datetime

def get_logger(config, level='INFO'):
    logger = logging.getLogger()
    logger.handlers = []
    now = datetime.datetime.now()
    logger_name = "logs/{}_{}transitions_{}_{}_{}.log".format(str(now.day) + str(now.hour)+ str(now.minute),config["num_transitions"],config["num_sprites"],("all_sprite_mover"if config["all_sprite_mover"] else "one_sprite_mover" if config["one_sprite_mover"] else "select_move"),config["random_init_places"])
    file_log_handler = logging.FileHandler(logger_name)
    logger.addHandler(file_log_handler)
    logger.setLevel(level)
    return logger

def train_epoch(model,train_loader,optimizer):
    model.train()
    
    train_loss = 0 
    for batch_idx, batch in enumerate(train_loader):
        inputs, label = batch
        optimizer.zero_grad()
        outputs, _ = model(inputs)
        
        loss = torch.nn.functional.mse_loss(outputs,label)
        loss.backward()
        train_loss += loss.item()
        
        optimizer.step()

    return train_loss/len(train_loader)

def validation_epoch(model,validation_loader):
    model.eval()
    
    val_loss = 0 
    for batch_idx, batch in enumerate(validation_loader):
        inputs, label = batch
        outputs, _ = model(inputs)
        loss = torch.nn.functional.mse_loss(outputs,label)
        val_loss += loss.item()

    
    return val_loss/len(validation_loader)        

def train(config):
    set_random_seed(config["seed"])
    
    data_name = "transition_data/{}transitions_{}_{}_{}.npz".format(config["num_transitions"],config["num_sprites"],("all_sprite_mover"if config["all_sprite_mover"] else "one_sprite_mover" if config["one_sprite_mover"] else "select_move"),config["random_init_places"])
    dataset = np.load(data_name)
    state_actions = np.concatenate((dataset["states"],dataset["actions"]),axis=1)
    tensor_dataset = TensorDataset(torch.from_numpy(state_actions).float(),torch.from_numpy(dataset["next_states"]).float())
    train_dataset, val_dataset = torch.utils.data.random_split(tensor_dataset, [int(len(tensor_dataset)*0.8), len(tensor_dataset)-int(len(tensor_dataset)*0.8)])

    train_dataloader = DataLoader(train_dataset,batch_size=config["batch_size"],shuffle=True)
    val_dataloader = DataLoader(val_dataset,batch_size=config["batch_size"],shuffle=False)
    
    if config["obj_disentanglement"]:
        obj_disentanglement = []
        for i in range(config["num_sprites"]):
            obj_disentanglement.append(list(range(i*4,(i+1)*4)))
        for j in range(config["num_sprites"]):
            obj_disentanglement.append(list(range((i+1)*4 + j*2,(i+1)*4 + (j+1)*2)))
    else:
        obj_disentanglement = None
        
    model = DynamicsModel(in_features=state_actions.shape[1],out_features=state_actions.shape[1],
                          state_dim=config["num_sprites"]*4,action_dim=config["num_sprites"]*2,
                          num_heads=config["num_heads"],
                          emb_dim=config["emb_dim"], object_disentanglement=obj_disentanglement,
                         num_attention_layers=config["num_attention_layers"],num_emb_layers=config["num_emb_layers"])
    #model = SimpleStackedAttn(in_features=state_actions.shape[1],out_features=state_actions.shape[1],)
    optimizer = torch.optim.Adam(model.parameters(),lr=config["lr"])
    
    bar = Bar('{}'.format('Training'), max=config["num_epochs"])
    logger = None
    if config["log"]:
        logger = get_logger(config)
        
    lowest_val_loss = 100000
    for epoch in (range(config["num_epochs"])):
        train_loss = train_epoch(model,train_dataloader,optimizer)
        val_loss = validation_epoch(model,val_dataloader)
        
        epoch_info = "Epoch: {} Train Loss: {} Val Loss: {}".format(epoch,train_loss,val_loss)
        if lowest_val_loss > val_loss:
            lowest_val_loss = val_loss
            epoch_info += "  Saved Model"
            torch.save(model.state_dict(), "models/{}transitions_{}_{}_{}.pt".format(config["num_transitions"],config["num_sprites"],("all_sprite_mover"if config["all_sprite_mover"] else "one_sprite_mover" if config["one_sprite_mover"] else "select_move"),config["random_init_places"]))
        
        
        if logger is not None:
            logger.info(epoch_info)
            
        Bar.suffix = epoch_info
        bar.next()
    
    
    bar.finish()
    print("Training Complete")
        
def main(config):
    
    train(config)    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--emb_dim', type=int, default=128)
    parser.add_argument('--num_heads', type=int, default=1)
    parser.add_argument('--num_attention_layers', type=int, default=2)
    parser.add_argument('--num_emb_layers', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_sprites', type=int, default=1)
    parser.add_argument('--one_sprite_mover', action="store_true")
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--all_sprite_mover', action="store_true")
    parser.add_argument('--num_transitions', type=int,default=20000)
    parser.add_argument('--random_init_places', action="store_true")
    parser.add_argument('--seed', type=int,default=0)
    parser.add_argument('--log', action="store_true")
    parser.add_argument('--obj_disentanglement', action="store_true")

    main(vars(parser.parse_args()))
    