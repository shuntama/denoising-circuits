import os
from Diffusion.Train import train, eval


def main(model_config = None):
    modelConfig = {
        # main configs
        "state": "train", # train or eval
        "model_type": "rnet", # "unet" or "rnet"
        "pred_mode": "v",  # "eps" or "v" or "x0"
        "use_x0_aux": True,  # when pred_mode=="v" or "eps": enable auxiliary x0 supervision and inference head
        "use_teacher_ddim": True,  # k>0: DDIM teacher supervision
        "enable_teacher_ema": False,  # if True and use_teacher_ddim: use separate EMA teacher model
        "epoch": 100,
        "batch_size": 128,
        "lr": 2e-4,
        "training_load_weight": None,  # None for start from scratch
        "test_load_weight": "ckpt_0100_ema.pt",
        "ring_infer": True,  # if True: use Rin→R×T path for inference
        "use_compile": True,  # torch.compile for faster training
        # hyperparameters
        "channel": 128,
        "lambda_consist": 5.0,  # consistency loss weight
        "lambda_x0_aux": 1.0,  # auxiliary x0 loss weight
        "step_probs": [0.0, 1.0],  # multi-step training (probabilities for s=0..K; None to disable)
        # sub configs
        "T": 1000,
        "beta_1": 1e-4,
        "beta_T": 0.02,
        "img_size": 32,
        "grad_clip": 1.,
        "ema_decay": 0.999,
        "device": "cuda:0",
        "save_weight_dir": "./Checkpoints/diffusion/",
        "sampled_dir": "./SampledImgs/diffusion/",
        "sampledNoisyImgName": "noisy.png",
        "sampledImgName": "sampled.png",
        "nrow": 8,
        }
    if model_config is not None:
        modelConfig = model_config
    
    # create necessary directories if they don't exist
    os.makedirs(modelConfig["save_weight_dir"], exist_ok=True)
    os.makedirs(modelConfig["sampled_dir"], exist_ok=True)
    os.makedirs("./CIFAR10/", exist_ok=True)
    
    if modelConfig["state"] == "train":
        train(modelConfig)
    else:
        eval(modelConfig)


if __name__ == '__main__':
    main()
