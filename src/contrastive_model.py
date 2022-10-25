import argparse
from stable_baselines3.common.utils import set_random_seed
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch
from progress.bar import Bar
import logging
import datetime
from datasets import TransitionDataset, RecursiveTransitionDataset
from networks import *
from utils import get_augmentation, soft_update_params,count_parameters, find_weight_norm
import os
import copy

class ContrastiveModel:
    def __init__(self, 
                encoder_model_name,
                action_dim,
                augmentations = ["shift", "intensity"],
                target_augmentations= ["shift", "intensity"],
                val_augmentations = "none",
                aug_prob=1,
                lr=0.0001,
                target_tau=0,
                device="cpu",
                seed=0):
        
        self.encoder_model_name = encoder_model_name
        self.device = device
        self.aug_prob = aug_prob
        self.transforms = get_augmentation(augmentations,84)
        self.target_transforms = get_augmentation(target_augmentations, 84)
        self.val_transforms = get_augmentation(val_augmentations, 84)
        self.target_tau = target_tau #tau should be bigger than when no augmentation is used
        set_random_seed(seed)
        
        now = datetime.datetime.now()
        self.exp_name = "{}_{}".format(str(now.day) + str(now.hour)+ str(now.minute),self.encoder_model_name)
        
        cnn_scale_factor = 1
        if encoder_model_name == "simple_cnn":
            
            self.encoder = Conv2dModel(in_channels=9,channels=[int(32*cnn_scale_factor),
                            int(64*cnn_scale_factor),
                            int(64*cnn_scale_factor)],
                    kernel_sizes=[8, 4, 3],
                    strides=[4, 2, 1],
                    paddings=[0, 0, 0],
                    use_maxpool=False,
                    dropout=0).to(self.device)
        elif encoder_model_name == "resnet":
            resblock = "inverted"
            
            if resblock == "inverted":
                resblock = InvertedResidual
            else:
                resblock = Residual

            self.encoder = ResnetCNN(9,
                        depths=[int(32*cnn_scale_factor),
                                int(64*cnn_scale_factor),
                                int(64*cnn_scale_factor)],
                        strides=[3, 2, 2],
                        norm_type="bn",
                        blocks_per_group=3,
                        resblock=resblock,
                        expand_ratio=3,).to(self.device)

        print("Initialized model with  parameters; CNN has {}.".format(count_parameters(self.encoder)))
        print("Initialized CNN weight norm is {}".format(find_weight_norm(self.encoder.parameters()).item()))

        self.target_encoder = copy.deepcopy(self.encoder).to(self.device)
        encoder_output_dim = 3136#self.encoder(dataset[0][0][0].unsqueeze(0).float().to(self.device)).numel()
        
        projection_dim = 256
        self.projection = nn.Sequential(
                        nn.Linear(encoder_output_dim, 512),
                        TransposedBN1D(512),
                        nn.ReLU(),
                        nn.Linear(512, projection_dim)
                        ).to(self.device)
        self.target_projection = copy.deepcopy(self.projection).to(self.device)
        print("Encoder output dim: ",encoder_output_dim)
        self.forward_model = MLP(projection_dim + action_dim, 256, projection_dim, 3).to(self.device)
        self.inverse_model = MLP(2 * projection_dim, 256, action_dim, 3).to(self.device)
                
        self.predictor = nn.Linear(projection_dim, projection_dim).to(self.device)
        
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(),lr=lr)
        self.forward_optimizer = torch.optim.Adam(self.forward_model.parameters(),lr=lr)
        self.inverse_optimizer = torch.optim.Adam(self.inverse_model.parameters(),lr=lr)
        self.predictor_optimizer = torch.optim.Adam(self.predictor.parameters(),lr=lr)

        self.logger = self.get_logger()
        
        
    def get_logger(self, level='INFO'):
        logger = logging.getLogger()
        logger.handlers = []
        
        logger_name = "logs/" + self.exp_name + ".log"
        file_log_handler = logging.FileHandler(logger_name)
        logger.addHandler(file_log_handler)
        logger.setLevel(level)
        return logger

    
    def train(self,train_dataloader, val_dataloader, num_epochs):
        
        bar = Bar('{}'.format('Training'), max=num_epochs)
        min_val_loss = 100000
        for epoch in range(num_epochs):
            spr_train_loss,inverse_train_loss = self.train_epoch(train_dataloader)
            spr_val_loss, inverse_val_loss = self.val_epoch(val_dataloader)         
            short_epoch_info = "Epoch: {},  train Loss: {}, Val Loss: {}".format(epoch,spr_train_loss+inverse_train_loss,spr_val_loss +inverse_val_loss)   
            
            epoch_info = "Epoch: {},TRAIN : SPR Loss: {}, Inverse loss: {} ||   VAL : SPR Loss: {}, Inverse Loss {}".format(epoch,spr_train_loss,inverse_train_loss, spr_val_loss, inverse_val_loss)
            self.logger.info(epoch_info)
            val_loss = spr_val_loss + inverse_val_loss
            if min_val_loss > val_loss:
                min_val_loss = val_loss
                self.save_models()
    
            Bar.suffix = short_epoch_info
            bar.next()
        bar.finish()

    def save_models(self):
        if not os.path.isdir("models"):
            os.mkdir("models")
        if not os.path.isdir("models/" + self.exp_name):
            os.mkdir("models/" + self.exp_name)
            
        torch.save(self.encoder.state_dict(), "models/{}/encoder.pt".format(self.exp_name))
        torch.save(self.forward_model.state_dict(), "models/{}/forward_model.pt".format(self.exp_name))
        torch.save(self.inverse_model.state_dict(), "models/{}/inverse_model.pt".format(self.exp_name))


    def project(self, x,target=False):
        if target:
            x = self.target_encoder(x)
            x = renormalize(x)
            x = x.view(x.size(0), -1)
            x = self.target_projection(x)
            return x
        else:
            x = self.encoder(x)
            x = renormalize(x)
            x = x.view(x.shape[0], -1)
            x = self.projection(x)
            return x
    
    
    def apply_transforms(self, transforms, image):
        for transform in transforms:
            image = maybe_transform(image, transform, p=self.aug_prob)
        return image

    @torch.no_grad()
    def transform(self, images, transforms, augment=False):
        images = ((images.float()/255.)-0.5) if images.dtype == torch.uint8 else images
        if augment:
            flat_images = images.reshape(-1, *images.shape[-3:])
            processed_images = self.apply_transforms(transforms,
                                                     flat_images)
            processed_images = processed_images.view(*images.shape[:-3],
                                                     *processed_images.shape[1:])
            return processed_images
        else:
            return images

    def update_targets(self):
        soft_update_params(self.encoder, self.target_encoder, self.target_tau)
        soft_update_params(self.projection, self.target_projection, self.target_tau)
        

    def normalized_l2_loss(self, x, y):
        
        f_x1 = F.normalize(x, p=2., dim=-1, eps=1e-3)
        f_x2 = F.normalize(y, p=2., dim=-1, eps=1e-3)
        return F.mse_loss(f_x1, f_x2)
    
    
    def do_spr_loss(self,proj_latents, targets):
        pred_latents = self.predictor(proj_latents)
        return self.normalized_l2_loss(pred_latents, targets)
        
    def do_inverse_loss(self,proj_latents,target_proj,actions):
        obs_pairs = torch.cat([proj_latents[:,:-1], target_proj[:,1:]], dim=2)
        
        pred_actions = self.inverse_model(obs_pairs.view(obs_pairs.shape[0]*obs_pairs.shape[1],-1))
        
        pred_actions = pred_actions.view(obs_pairs.shape[0],obs_pairs.shape[1],-1)
        
        return F.mse_loss(pred_actions, actions[:,:-1],-1)


    def process_recursive_batch(self, batch):
    
        obs, action, next_obs = batch
        
        obs = obs.to(self.device)
        action = action.to(self.device)
        
        first_obs = self.transform(obs[:,0], self.transforms, augment=True)   
        
        latent = self.encoder(first_obs)
        latent = renormalize(latent)
        latent = self.projection(latent.view(latent.shape[0], -1))
        pred_latents = [latent]
        
        for step in range(1,obs.shape[1]):
            latent = self.forward_model(torch.cat([latent, action[:,step]], dim=1))
            pred_latents.append(latent)
                
        proj_latents = torch.stack(pred_latents, dim=1)
        #proj_latents = self.projection(pred_latents.view(pred_latents.shape[0]*pred_latents.shape[1], -1))
        with torch.no_grad():
            target_obs = self.transform(obs, self.transforms, augment=True)
            target_proj = self.project(target_obs.view(target_obs.shape[0]*target_obs.shape[1],*target_obs.shape[2:]), target=True)
            target_proj = target_proj.view(target_obs.shape[0],target_obs.shape[1],-1)
        # SPR LOSS
        spr_loss = self.do_spr_loss(proj_latents, target_proj)
        
        # INVERSE LOSS
        inverse_loss = self.do_inverse_loss(proj_latents, target_proj,action)
        
        return spr_loss, inverse_loss

        
    def train_epoch(self,train_dataloader):
        self.encoder.train()
        self.forward_model.train()
        self.inverse_model.train()
        
        epoch_spr_loss = 0
        epoch_inverse_loss = 0
        for batch_idx, batch in enumerate(train_dataloader):
            
            spr_loss, inverse_loss = self.process_recursive_batch(batch)
            loss = (10 * spr_loss) + inverse_loss
            loss.backward()
            
            epoch_spr_loss += spr_loss.item()
            epoch_inverse_loss += inverse_loss.item()
            
            self.encoder_optimizer.step()
            self.forward_optimizer.step()
            self.inverse_optimizer.step()
            self.predictor_optimizer.step()
            self.update_targets()
            
        return epoch_spr_loss/len(self.train_dataloader), epoch_inverse_loss/len(self.train_dataloader)

    def val_epoch(self,val_dataloader):
        self.encoder.eval()
        self.forward_model.eval()
        self.inverse_model.eval()
        
        epoch_spr_loss = 0
        epoch_inverse_loss = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataloader):
                
                spr_loss, inverse_loss = self.process_recursive_batch(batch)
                
                epoch_spr_loss += spr_loss.item()
                epoch_inverse_loss += inverse_loss.item()

        return epoch_spr_loss/len(self.val_dataloader), epoch_inverse_loss/len(self.val_dataloader)
    
    def encode_obs(self,obs):
        with torch.no_grad():
            obs = self.transform(obs, self.val_transforms, augment=False)
            latent = self.encoder(obs)
            latent = renormalize(latent)
            return self.projection(latent.view(latent.shape[0], -1))
        
        
    def process_batch(self, batch):
    
        obs, action, next_obs = batch
        
        obs = obs.to(self.device)
        next_obs = next_obs.to(self.device)
        action = action.to(self.device)
        
        obs_next_obs = torch.cat([obs, next_obs], dim=1)
        obs_next_obs = self.transform(obs_next_obs, self.transforms, augment=True)
        obs = obs_next_obs[:, :obs.shape[1]]
        next_obs = obs_next_obs[:, obs.shape[1]:]
        
        state_embedding = self.project(obs)
        state_target_embedding = self.project(obs,target=True)
        
        # SPR LOSS
        spr_latent = self.predictor(state_embedding)
        spr_loss = self.normalized_l2_loss(spr_latent, state_target_embedding)

        # INVERSE LOSS
        next_state_target_embedding = self.project(next_obs,target=True)
        
        pred_action = self.inverse_model(torch.cat([state_embedding, next_state_target_embedding], dim=1))
        inverse_loss = F.mse_loss(pred_action, action)
        
        loss = spr_loss + inverse_loss
            
        return loss


def main(config):
    
    print("Training with config: {}".format(config))
    
    assert config["replay_dir"] is not None, "Replay directory is not specified"
            
    model = ContrastiveModel(**config)
    model.train()
    print("Training Done")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--seed', type=int,default=0)
    parser.add_argument('--encoder_model_name', type=str, default="simple_cnn")
    parser.add_argument('--replay_dir', type=str, default=None)
    parser.add_argument('--device', type=str, default="cpu")
    parser.add_argument('--num_workers',type=int, default=0)

    main(vars(parser.parse_args()))
    
