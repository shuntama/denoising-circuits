import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


def _cosine_blend_losses(loss_fine: torch.Tensor, loss_coarse: torch.Tensor, t_eff: torch.Tensor, T: int) -> torch.Tensor:
    """
    Blend two per-example losses with a cosine schedule over time:
      - t_eff = 0   -> weight_fine = 1, weight_coarse = 0
      - t_eff = T-1 -> weight_fine = 0, weight_coarse = 1
    """
    tau = t_eff.float() / max(T - 1, 1)
    weight_coarse = 0.5 * (1.0 - torch.cos(math.pi * tau))
    weight_fine = 1.0 - weight_coarse
    if weight_coarse.ndim == 1:
        weight_coarse = weight_coarse.view(-1, 1, 1, 1)
        weight_fine = weight_fine.view(-1, 1, 1, 1)
    return weight_fine * loss_fine + weight_coarse * loss_coarse


class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, step_probs=None, pred_mode: str = "eps", use_x0_aux: bool = False, use_teacher_ddim: bool = False, teacher_model: nn.Module | None = None):
        super().__init__()

        self.model = model
        self.T = T
        self.pred_mode = str(pred_mode)
        self.use_x0_aux = bool(use_x0_aux)
        self.use_teacher_ddim = bool(use_teacher_ddim)
        self.teacher_model = teacher_model

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).float())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer('sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

        # whether model exposes required submodules (RNet: rin, r)
        self.has_r = (
            hasattr(self.model, 'rin') and hasattr(self.model, 'r')
        )
        # Optional multi-step (1..K steps) probabilities (batch-shared)
        if step_probs is not None:
            sp = torch.tensor(step_probs, dtype=torch.float32)
            sp = sp / (sp.sum() + 1e-12)
            # allow arbitrary length K; interpret indices 0..K-1 as s=1..K
            self.register_buffer('step_probs', sp)
        else:
            self.step_probs = None

    def forward(self, x_0):
        """
        Algorithm 1 + optional MSE consistency on features: R(Rin(x_t)) vs Rin(x_{t-1}).
        Returns: (main_loss_tensor, consist_mse_tensor)
        """
        B = x_0.shape[0]

        # Non-R path (e.g., UNet) or R path without multi-step training
        if not self.has_r or self.step_probs is None:
            t = torch.randint(self.T, size=(B, ), device=x_0.device)
            noise = torch.randn_like(x_0)
            x_t = (
                extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
                extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
            pred = self.model(x_t, t)
            if self.pred_mode == 'v':
                c0 = extract(self.sqrt_alphas_bar, t, x_0.shape)
                c1 = extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape)
                v_target = c0 * noise - c1 * x_0
                main_loss_tensor = F.mse_loss(pred, v_target, reduction='none')
            elif self.pred_mode == 'x0':
                main_loss_tensor = F.mse_loss(pred, x_0, reduction='none')
            else:
                main_loss_tensor = F.mse_loss(pred, noise, reduction='none')
            zeros = torch.zeros_like(main_loss_tensor)
            return main_loss_tensor, zeros, zeros

        # R path (multi-step training compatible)
        # choose number of steps s in {0..K-1} from provided distribution (batch-shared)
        s = 0
        if self.step_probs is not None:
            s_idx = torch.multinomial(self.step_probs, 1).item()  # 0..K-1 (0 corresponds to s=0)
            s = int(s_idx)
        # sample t with lower bound; ensure t>=s
        t_low = s
        t = torch.randint(self.T - t_low, size=(B, ), device=x_0.device) + t_low
        # sample noise for x_t
        noise = torch.randn_like(x_0)
        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        
        # prediction via R
        if self.has_r and self.step_probs is not None:
            # Rin once; then R for every step k=0..s and compute losses
            z0 = self.model.rin(x_t, t)
            z = z0
            main_loss_0 = None  # L0: loss at k=0
            main_loss_sum = 0.0  # Ls: sum of losses at k>=1
            x0_aux_loss_0 = None  # L0: x0 aux loss at k=0
            x0_aux_loss_sum = 0.0  # Ls: sum of x0 aux losses at k>=1
            consist_mse_sum = 0.0
            
            # optional: build teacher chain x_teacher_list[k] via DDIM deterministic update for k=0..s
            x_teacher_list = None
            if self.use_teacher_ddim and (self.teacher_model is not None):
                with torch.no_grad():
                    self.teacher_model.eval()
                    x_teacher_list = [x_t]
                    z_teacher_list = []  # length s+1, aligned with k=0..s: [Rin(x_t), Rin(x_{t-1}), ...]
                    pred_teacher_list = []
                    for step in range(1, int(s) + 1):
                        t_curr = t - (step - 1)  # corresponds to k = step-1
                        x_curr = x_teacher_list[step - 1]
                        # split teacher inference into Rin and Rout
                        z_curr = self.teacher_model.rin(x_curr, t_curr)
                        pred_curr = self.teacher_model.rout(z_curr)
                        z_teacher_list.append(z_curr)
                        pred_teacher_list.append(pred_curr)
                        # derive eps from teacher pred by current mode
                        c0_curr = extract(self.sqrt_alphas_bar, t_curr, x_curr.shape)
                        c1_curr = extract(self.sqrt_one_minus_alphas_bar, t_curr, x_curr.shape)
                        if self.pred_mode == 'v':
                            eps_t = c1_curr * x_curr + c0_curr * pred_curr
                        elif self.pred_mode == 'x0':
                            eps_t = (x_curr - c0_curr * pred_curr) / (c1_curr + 1e-12)
                        else:
                            eps_t = pred_curr
                        # DDIM deterministic update (eta=0): x0_pred -> x_{t-1}
                        x0_pred = (x_curr - c1_curr * eps_t) / (c0_curr + 1e-12)
                        t_prev = t_curr - 1
                        c0_prev = extract(self.sqrt_alphas_bar, t_prev, x_curr.shape)
                        c1_prev = extract(self.sqrt_one_minus_alphas_bar, t_prev, x_curr.shape)
                        x_prev = c0_prev * x0_pred + c1_prev * eps_t
                        x_teacher_list.append(x_prev)
                    # finalize k=s entries
                    t_eff_s = t - s
                    x_s = x_teacher_list[s]
                    z_s = self.teacher_model.rin(x_s, t_eff_s)
                    pred_s = self.teacher_model.rout(z_s)
                    z_teacher_list.append(z_s)
                    pred_teacher_list.append(pred_s)
                self.teacher_model.train()
            
            for k in range(int(s) + 1):
                # Rout at effective timestep t_eff = t - k
                t_eff_k = t - k
                pred_k = self.model.rout(z)
                # main loss at step k
                if self.pred_mode == 'v':
                    if (k == 0) or (not self.use_teacher_ddim) or (x_teacher_list is None):
                        c0_k = extract(self.sqrt_alphas_bar, t_eff_k, x_0.shape)
                        c1_k = extract(self.sqrt_one_minus_alphas_bar, t_eff_k, x_0.shape)
                        v_target_k = c0_k * noise - c1_k * x_0
                    else:
                        v_target_k = pred_teacher_list[k]  # = self.teacher_model(x_teacher_list[k], t_eff_k)
                    main_loss_k = F.mse_loss(pred_k, v_target_k, reduction='none')
                elif self.pred_mode == 'x0':
                    if (k == 0) or (not self.use_teacher_ddim) or (x_teacher_list is None):
                        x0_target_k = x_0
                    else:
                        x0_target_k = pred_teacher_list[k]  # = self.teacher_model(x_teacher_list[k], t_eff_k)
                    main_loss_k = F.mse_loss(pred_k, x0_target_k, reduction='none')
                else:
                    if (k == 0) or (not self.use_teacher_ddim) or (x_teacher_list is None):
                        eps_target_k = noise
                    else:
                        eps_target_k = pred_teacher_list[k]  # = self.teacher_model(x_teacher_list[k], t_eff_k)
                    main_loss_k = F.mse_loss(pred_k, eps_target_k, reduction='none')
                if k == 0:
                    main_loss_0 = main_loss_k
                else:
                    main_loss_sum = main_loss_sum + main_loss_k
                # optional auxiliary x0 supervision via x0out
                if self.use_x0_aux and (self.pred_mode == 'v' or self.pred_mode == 'eps'):
                    x0_pred_k = self.model.x0out(z)
                    if (k == 0) or (not self.use_teacher_ddim) or (x_teacher_list is None):
                        x0_target_k = x_0
                    else:
                        z_teacher_k = z_teacher_list[k]  # = self.teacher_model.rin(x_teacher_list[k], t_eff_k) 
                        with torch.no_grad():
                            x0_target_k = self.teacher_model.x0out(z_teacher_k)
                    x0_loss_k = F.mse_loss(x0_pred_k, x0_target_k, reduction='none')
                    if k == 0:
                        x0_aux_loss_0 = x0_loss_k
                    else:
                        x0_aux_loss_sum = x0_aux_loss_sum + x0_loss_k
                # consistency MSE at step k>=1: R^k(Rin(x_t)) vs Rin(x_{t-k})
                if k >= 1:
                    z_R_k = z  # current ring output after k laps: R^k(Rin(x_t))
                    if self.use_teacher_ddim and (z_teacher_list is not None):
                        z_tm_k = z_teacher_list[k]  # = self.teacher_model.rin(x_teacher_list[k], t_eff_k)
                    else:
                        x_tk = (
                            extract(self.sqrt_alphas_bar, t_eff_k, x_0.shape) * x_0 +
                            extract(self.sqrt_one_minus_alphas_bar, t_eff_k, x_0.shape) * noise
                        )
                        z_tm_k = self.model.rin(x_tk.detach(), t_eff_k)

                    # consistency MSE with nested cosine transitions over multi-scale pooling
                    # level 0: full-resolution MSE
                    consist_mse_0 = F.mse_loss(z_R_k, z_tm_k, reduction='none')
                    consist_mse_0 = consist_mse_0.mean(dim=[1, 2, 3], keepdim=True)
                    # level 1: AvgPool(2)
                    pooled_z_R_k = F.avg_pool2d(z_R_k, kernel_size=2, stride=2)
                    pooled_z_tm_k = F.avg_pool2d(z_tm_k, kernel_size=2, stride=2)
                    consist_mse_1 = F.mse_loss(pooled_z_R_k, pooled_z_tm_k, reduction='none')
                    consist_mse_1 = consist_mse_1.mean(dim=[1, 2, 3], keepdim=True)
                    # level 2: further AvgPool(2) on level-1 features
                    pooled_z_R_k = F.avg_pool2d(pooled_z_R_k, kernel_size=2, stride=2)
                    pooled_z_tm_k = F.avg_pool2d(pooled_z_tm_k, kernel_size=2, stride=2)
                    consist_mse_2 = F.mse_loss(pooled_z_R_k, pooled_z_tm_k, reduction='none')
                    consist_mse_2 = consist_mse_2.mean(dim=[1, 2, 3], keepdim=True)
                    # first blend: level 1 vs level 2
                    consist_mse_12 = _cosine_blend_losses(consist_mse_1, consist_mse_2, t_eff_k, self.T)
                    # second blend: level 0 vs level 1
                    consist_mse_01 = _cosine_blend_losses(consist_mse_0, consist_mse_12, t_eff_k, self.T)
                    #consist_mse_sum = consist_mse_sum + consist_mse_01
                    consist_mse_sum = consist_mse_sum + consist_mse_0  # disable cosine blending for debugging
                
                # advance R once for next step (unless reached s)
                if k < s:
                    z = self.model.r.forward_one_step(z, t_eff_k)
            
            # compute final losses: {L(s=0) + w * ΣL(s>0) / s} / (1.0 + w)
            w = 1.0  # loss weight for k>=1 steps
            if s == 0:
                main_loss_tensor = main_loss_0
                if self.use_x0_aux:
                    x0_aux_loss_tensor = x0_aux_loss_0
                else:
                    x0_aux_loss_tensor = torch.zeros_like(main_loss_tensor)
                consist_mse_tensor = torch.zeros_like(main_loss_tensor)
            else:
                main_loss_tensor = (main_loss_0 + w * main_loss_sum / float(s)) / (1.0 + w)
                if self.use_x0_aux:
                    x0_aux_loss_tensor = (x0_aux_loss_0 + w * x0_aux_loss_sum / float(s)) / (1.0 + w)
                else:
                    x0_aux_loss_tensor = torch.zeros_like(main_loss_tensor)
                consist_mse_tensor = consist_mse_sum / float(s)

        return main_loss_tensor, consist_mse_tensor, x0_aux_loss_tensor


class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, pred_mode: str = "eps", ring_infer: bool = False, use_x0_aux: bool = False):
        super().__init__()

        self.model = model
        self.T = T
        self.pred_mode = str(pred_mode)
        self.ring_infer = bool(ring_infer)
        self.use_x0_aux = bool(use_x0_aux)

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).float())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))

        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.coeff1, t, x_t.shape) * x_t -
            extract(self.coeff2, t, x_t.shape) * eps
        )

    def p_mean_variance(self, x_t, t):
        # below: only log_variance is used in the KL computations
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var, t, x_t.shape)

        pred = self.model(x_t, t)
        # derive eps_pred from pred by mode
        if self.pred_mode == 'v':
            c0 = extract(torch.sqrt(torch.cumprod(1. - self.betas, dim=0)), t, x_t.shape)
            c1 = extract(torch.sqrt(1. - torch.cumprod(1. - self.betas, dim=0)), t, x_t.shape)
            eps = c1 * x_t + c0 * pred
        elif self.pred_mode == 'x0':
            c0 = extract(torch.sqrt(torch.cumprod(1. - self.betas, dim=0)), t, x_t.shape)
            c1 = extract(torch.sqrt(1. - torch.cumprod(1. - self.betas, dim=0)), t, x_t.shape)
            eps = (x_t - c0 * pred) / (c1 + 1e-12)
        else:
            eps = pred
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)

        return xt_prev_mean, var

    def forward(self, x_T):
        """
        Algorithm 2.
        """
        # special path: R-only inference (no DDPM updates)
        # - x0 mode: use Rout
        # - v/eps mode with x0_aux enabled: use X0out
        if self.ring_infer:
            B = x_T.shape[0]
            t_init = x_T.new_ones([B, ], dtype=torch.long) * (self.T - 1)
            z = self.model.rin(x_T, t_init)
            for time_step in reversed(range(self.T)):
                if time_step % 100 == 0:
                    print(time_step)
                t = x_T.new_ones([B, ], dtype=torch.long) * time_step
                z = self.model.r.forward_one_step(z, t)
            # Prefer x0_aux head if enabled
            if self.use_x0_aux:
                x0_pred = self.model.x0out(z)
            elif self.pred_mode == 'x0':
                x0_pred = self.model.rout(z)
            elif self.pred_mode == 'v' or self.pred_mode == 'eps':
                # Convert Rout prediction (v or eps) into x0 using one-step formulas at t=T-1
                pred = self.model.rout(z)
                t0 = x_T.new_ones([B, ], dtype=torch.long) * (self.T - 1)
                c0 = extract(torch.sqrt(torch.cumprod(1. - self.betas, dim=0)), t0, x_T.shape)
                c1 = extract(torch.sqrt(1. - torch.cumprod(1. - self.betas, dim=0)), t0, x_T.shape)
                if self.pred_mode == 'v':
                    # x0 = c0 * x_t - c1 * v, evaluated at t=T-1 with x_t=x_T
                    x0_pred = c0 * x_T - c1 * pred
                else:
                    # x0 = (x_t - c1 * eps) / c0, evaluated at t=T-1 with x_t=x_T
                    x0_pred = (x_T - c1 * pred) / (c0 + 1e-12)
            else:
                raise ValueError("Invalid prediction mode for special ring_infer path.")
            return torch.clip(x0_pred, -1, 1)

        x_t = x_T
        for time_step in reversed(range(self.T)):
            if time_step % 100 == 0:
                print(time_step)
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            mean, var= self.p_mean_variance(x_t=x_t, t=t)
            # no noise when t == 0
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            x_t = mean + torch.sqrt(var) * noise
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
        x_0 = x_t
        return torch.clip(x_0, -1, 1)   


