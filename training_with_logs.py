import argparse
import torch
import torch.optim as optim
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import csv
from datetime import datetime
from pathlib import Path

from src.data.loader import get_dataloaders
from src.models.network import ESCAN
from src.utils.loss import CompositeFusionLoss

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class RunLogger:
    def __init__(self, config, args):
        timestamp = datetime.now().strftime('%b%d_%H-%M')
        
        l1 = config['loss']['lambda_1']
        l2 = config['loss']['lambda_2']
        l3 = config['loss']['lambda_3']
        run_name = f"ESCAN_{timestamp}_L1-{l1}_L2-{l2}_L3-{l3}"
        
        self.run_dir = Path(config['log_dir']) / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        with open(self.run_dir / "config_used.yaml", "w") as f:
            yaml.dump(config, f)
            
        self.csv_path = self.run_dir / "training_log.csv"
        
        self.headers = ["Epoch", "Train_Loss", "Val_Loss", "L_int", "L_ssim", "L_grad", "LR"]
        with open(self.csv_path, "w", newline="") as f:
            csv.writer(f).writerow(self.headers)

    def log_epoch(self, epoch, stats):
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, 
                stats['train_loss'], 
                stats['val_loss'], 
                stats['l_int'], 
                stats['l_ssim'], 
                stats['l_grad'], 
                stats['lr']
            ])
            
    def get_ckpt_path(self, epoch):
        return self.run_dir / f"escan_epoch_{epoch}.pth"
        
    def get_best_ckpt_path(self):
        return self.run_dir / "escan_best_model.pth"

def train(args):
    # 1. Load Config
    config = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 2. Setup Logger
    logger = RunLogger(config, args)
    writer = SummaryWriter(log_dir=str(logger.run_dir))
    print(f"Starting ESCAN Training Run: {logger.run_dir}")

    # 3. Data Loaders (Inputs are bounded [0, 1])
    train_loader, val_loader = get_dataloaders(config)
    
    # 4. Initialize ESCAN Model
    model = ESCAN(n_channels=32, init_type=config.get('initialization', 'Kaiming')).to(device)
    
    # 5. Optimizer & Loss 
    opt_cfg = config['optimizer']
    optimizer = optim.Adam(
        model.parameters(), 
        lr=float(opt_cfg['lr']), 
        weight_decay=float(opt_cfg['weight_decay'])
    )
    
    sch_cfg = config['scheduler']
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=sch_cfg['step_size'], 
        gamma=sch_cfg['gamma']
    )

    criterion = CompositeFusionLoss(
        lambda_1=config['loss']['lambda_1'],
        lambda_2=config['loss']['lambda_2'],
        lambda_3=config['loss']['lambda_3']
    ).to(device)

    epochs = config['epochs'] 
    
    # Tracking the best validation model
    best_val_loss = float('inf')
    
    # 6. Training Loop
    for epoch in range(1, epochs + 1):
        model.train()
        t_loss, t_int, t_ssim, t_grad = 0.0, 0.0, 0.0, 0.0
        
        # --- TRAIN ---
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
        for batch in pbar:
            i_mr = batch['mri'].to(device)
            i_ct = batch['ct'].to(device)
            
            optimizer.zero_grad()
            
            i_f = model(i_mr, i_ct)
            
            loss, l_int, l_ssim, l_grad = criterion(i_mr, i_ct, i_f)
            
            loss.backward()
            optimizer.step()
            
            t_loss += loss.item()
            t_int += l_int.item()
            t_ssim += l_ssim.item()
            t_grad += l_grad.item()
            
            pbar.set_postfix({'Total Loss': f"{loss.item():.4f}"})
            
        model.eval()
        v_loss = 0.0
        
        if len(val_loader) > 0:
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]", leave=False)
            with torch.no_grad():
                for batch in val_pbar:
                    i_mr = batch['mri'].to(device)
                    i_ct = batch['ct'].to(device)
                    
                    i_f = model(i_mr, i_ct)
                    loss, _, _, _ = criterion(i_mr, i_ct, i_f)
                    v_loss += loss.item()
        
        last_lr = scheduler.get_last_lr()[0]
        scheduler.step() 
        
        avg_train_loss = t_loss / len(train_loader)
        avg_val_loss = v_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        
        stats = {
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'l_int': t_int / len(train_loader),
            'l_ssim': t_ssim / len(train_loader),
            'l_grad': t_grad / len(train_loader),
            'lr': last_lr
        }
        
        logger.log_epoch(epoch, stats)
        
        writer.add_scalar('Loss/Train (Total)', stats['train_loss'], epoch)
        writer.add_scalar('Loss/Val (Total)', stats['val_loss'], epoch)
        writer.add_scalar('LossComponents/Intensity', stats['l_int'], epoch)
        writer.add_scalar('LossComponents/SSIM', stats['l_ssim'], epoch)
        writer.add_scalar('LossComponents/Gradient', stats['l_grad'], epoch)
        writer.add_scalar('Hyperparameters/LR', last_lr, epoch)
        

        torch.save(model.state_dict(), logger.get_ckpt_path(epoch))
        
        if len(val_loader) > 0 and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), logger.get_best_ckpt_path())
            print(f"  --> New Best Model saved! (Val Loss: {best_val_loss:.4f})")
            
    writer.close()
    print("\nESCAN Training Complete.")
    print(f"Best model saved at: {logger.get_best_ckpt_path()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    args = parser.parse_args()
    train(args)