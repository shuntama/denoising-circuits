import os
from typing import Dict

import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image

from Diffusion import GaussianDiffusionSampler, GaussianDiffusionTrainer
from Diffusion.ModelU import UNet
from Diffusion.ModelR import RNet


class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

    def copy_to(self, target_model):
        for name, param in target_model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name].to(device=param.data.device, dtype=param.data.dtype))


def train(modelConfig: Dict):
    # set up environment
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    device = torch.device(modelConfig["device"])
    
    # load dataset
    dataset = CIFAR10(
        root='./CIFAR10', train=True, download=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
    dataloader = DataLoader(
        dataset, batch_size=modelConfig["batch_size"], shuffle=True, num_workers=8, drop_last=True, pin_memory=True)

    # model setup
    if modelConfig["model_type"] == "rnet":
        net_model = RNet(T=modelConfig["T"], ch=modelConfig["channel"]).to(device)
        net_model = net_model.to(memory_format=torch.channels_last)
    else:
        net_model = UNet(T=modelConfig["T"]).to(device)
        net_model = net_model.to(memory_format=torch.channels_last)
    if modelConfig.get("use_compile", True):
        try:
            net_model = torch.compile(net_model, mode="reduce-overhead", fullgraph=False)  # torch.compile for network model
        except Exception:
            pass

    # load training weights
    if modelConfig["training_load_weight"] is not None:
        load_path = modelConfig["training_load_weight"]
        if not os.path.isabs(load_path):
            load_path = os.path.join(modelConfig["save_weight_dir"], load_path)
        ckpt = torch.load(load_path, map_location='cpu')
        model_to_load = getattr(net_model, '_orig_mod', net_model)
        model_to_load.load_state_dict(ckpt, strict=True)
    
    # teacher setup (for DDIM self-supervision)
    use_teacher_ddim = modelConfig.get("use_teacher_ddim", False)
    enable_teacher_ema = modelConfig.get("enable_teacher_ema", False)
    teacher_model = None
    if use_teacher_ddim:
        if enable_teacher_ema:
            # build a separate teacher model with the same architecture
            if modelConfig["model_type"] == "rnet":
                teacher_model = RNet(T=modelConfig["T"], ch=modelConfig["channel"])
            else:
                teacher_model = UNet(T=modelConfig["T"])
            # initialize teacher from current student weights (unwrap if compiled)
            student_base = getattr(net_model, "_orig_mod", net_model)
            teacher_model.load_state_dict(student_base.state_dict(), strict=True)
            teacher_model = teacher_model.to(device).to(memory_format=torch.channels_last)
            teacher_model.eval()
            for p in teacher_model.parameters():
                p.requires_grad = False
        else:
            # fallback: use student model directly (original behavior)
            teacher_model = net_model

    # optimizer and scheduler setup
    epoch_value = modelConfig["epoch"]
    is_fractional_epoch = (epoch_value < 1.0)
    if is_fractional_epoch:
        total_batches = int(epoch_value * len(dataloader))
        total_steps = total_batches
        print(f"Fractional epoch mode: {epoch_value} epoch = {total_batches} batches")
    else:
        total_batches = None
        total_steps = int(epoch_value) * len(dataloader)
    
    optimizer = torch.optim.AdamW(net_model.parameters(), weight_decay=1e-4, fused=True)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer, 
        max_lr=modelConfig["lr"], 
        total_steps=total_steps,
        pct_start=0.05,  # 5% of total steps for warmup
        anneal_strategy='cos',  # cosine annealing after warmup
        div_factor=10000.0,  # initial_lr = max_lr / div_factor
        final_div_factor=10000.0  # final_lr = max_lr / final_div_factor
    )

    # trainer setup
    trainer = GaussianDiffusionTrainer(
        net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"],
        step_probs=modelConfig.get("step_probs", None),
        pred_mode=modelConfig.get("pred_mode", "eps"),
        use_x0_aux=modelConfig.get("use_x0_aux", False),
        use_teacher_ddim=use_teacher_ddim,
        teacher_model=teacher_model,
    ).to(device)
    if modelConfig.get("use_compile", True):
        try:
            trainer = torch.compile(trainer, mode="reduce-overhead", fullgraph=False)  # torch.compile for trainer
        except Exception:
            pass
    
    # EMA setup
    ema_decay = modelConfig["ema_decay"]
    ema = EMA(net_model, decay=ema_decay)

    # AMP setup
    scaler = GradScaler()
    amp_dtype = torch.float16

    # start training
    num_epochs = 1 if is_fractional_epoch else int(epoch_value)
    for e in range(num_epochs):
        epoch_mse = 0.0
        epoch_consist_mse = 0.0
        epoch_x0_aux = 0.0
        num_batches = 0
        should_break = False
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for images, labels in tqdmDataLoader:
                # train
                optimizer.zero_grad(set_to_none=True)
                x_0 = images.to(device, non_blocking=True).to(memory_format=torch.channels_last)

                with autocast(dtype=amp_dtype):
                    eps_loss_tensor, consist_mse_tensor, x0_aux_loss_tensor = trainer(x_0)
                    loss = eps_loss_tensor.mean()
                epoch_mse += loss.item()
                # consistency MSE
                if modelConfig["model_type"] == "rnet":
                    with autocast(dtype=amp_dtype):
                        consist_mse_loss = consist_mse_tensor.mean()
                        loss = loss + modelConfig["lambda_consist"] * consist_mse_loss
                    epoch_consist_mse += consist_mse_loss.item()
                # auxiliary x0 loss
                if modelConfig.get("use_x0_aux", False):
                    with autocast(dtype=amp_dtype):
                        x0_aux_loss = x0_aux_loss_tensor.mean()
                        loss = loss + modelConfig["lambda_x0_aux"] * x0_aux_loss
                    epoch_x0_aux += x0_aux_loss.item()
                # backward pass
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(net_model.parameters(), modelConfig["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
                # update EMA after optimizer step
                ema.update()
                # keep EMA teacher in sync (if enabled)
                if use_teacher_ddim and enable_teacher_ema and (teacher_model is not None):
                    ema.copy_to(teacher_model)
                num_batches += 1
                # log progress
                postfix_dict = {
                    "epoch": e if not is_fractional_epoch else f"{epoch_value:.3f}",
                    "mse": f"{1e3 * epoch_mse/num_batches:.4f}",
                    "consist_mse": f"{1e3 * epoch_consist_mse/num_batches:.4f}",
                    "x0_loss": f"{1e3 * epoch_x0_aux/num_batches:.4f}",
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                }
                tqdmDataLoader.set_postfix(ordered_dict=postfix_dict)
                scheduler.step()
                
                # Check if fractional epoch should terminate
                if is_fractional_epoch and num_batches >= total_batches:
                    should_break = True
                    break

        # save checkpoint every 10 epochs and at the end of training
        if is_fractional_epoch:
            should_save = True
            epoch_label = f"{epoch_value:.3f}".replace('.', '_')
        else:
            should_save = ((e + 1) % 10 == 0 or (e + 1) == num_epochs)
            epoch_label = f"{e+1:04d}"
        
        if should_save:
            # save regular model (unwrap if compiled)
            model_to_save = getattr(net_model, '_orig_mod', net_model)
            torch.save(model_to_save.state_dict(), os.path.join(
                modelConfig["save_weight_dir"], f'ckpt_{epoch_label}.pt'))
            # save EMA model
            ema.apply_shadow()
            torch.save(model_to_save.state_dict(), os.path.join(
                modelConfig["save_weight_dir"], f'ckpt_{epoch_label}_ema.pt'))
            ema.restore()
        
        # break if fractional epoch terminated early
        if should_break:
            break


def eval(modelConfig: Dict):
    # load model and evaluate
    with torch.no_grad():
        device = torch.device(modelConfig["device"])
        if modelConfig["model_type"] == "rnet":
            model = RNet(T=modelConfig["T"], ch=modelConfig["channel"])
        else:
            model = UNet(T=modelConfig["T"])
        ckpt = torch.load(os.path.join(modelConfig["save_weight_dir"], modelConfig["test_load_weight"]), map_location='cpu')
        model.load_state_dict(ckpt, strict=True)
        print("model load weight done.")
        model = model.to(device).to(memory_format=torch.channels_last)
        model.eval()
        sampler = GaussianDiffusionSampler(
            model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"],
            pred_mode=modelConfig.get("pred_mode", "eps"),
            ring_infer=modelConfig.get("ring_infer", False),
            use_x0_aux=modelConfig.get("use_x0_aux", False),
        ).to(device)
        #try:
        #    sampler = torch.compile(sampler, mode="reduce-overhead", fullgraph=False)  # torch.compile for sampler
        #except Exception:
        #    pass
        # sampled from standard normal distribution
        noisyImage = torch.randn(size=[modelConfig["batch_size"], 3, 32, 32], device=device).to(memory_format=torch.channels_last)
        saveNoisy = torch.clamp(noisyImage * 0.5 + 0.5, 0, 1)
        save_image(saveNoisy, os.path.join(modelConfig["sampled_dir"], modelConfig["sampledNoisyImgName"]), nrow=modelConfig["nrow"])
        with autocast(dtype=torch.float16):
            sampledImgs = sampler(noisyImage)
        sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]
        save_image(sampledImgs, os.path.join(modelConfig["sampled_dir"],  modelConfig["sampledImgName"]), nrow=modelConfig["nrow"])