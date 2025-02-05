# %%writefile /kaggle/working/motion-diffusion-model/utils/model_util.py
from model.mdm import MDM
from model.mdm_unet import MDM_UNetModel
from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps
from utils.parser_util import get_cond_mode


def load_model_wo_clip(model, state_dict):
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)


def create_model_and_diffusion(args, data, num_joints=None):
    if args.arch == 'unet':
        # support_ = get_model_args(args, data)
        motion_args = get_unet_model_args(args, data, num_joints)
        model = create_motion_unet(
            motion_args,
            args.image_size,
            num_channels=256,
            num_res_blocks=8,
            channel_mult="1",
            learn_sigma=args.learn_sigma,
            class_cond=args.class_cond,
            use_checkpoint=args.use_checkpoint,
            attention_resolutions=args.attention_resolutions,
            num_heads=8,
            num_head_channels=args.num_head_channels,
            num_heads_upsample=args.num_heads_upsample,
            use_scale_shift_norm=True,
            dropout=0.0,
            resblock_updown=args.resblock_updown,
            use_fp16=args.use_fp16,
            use_new_attention_order=args.use_new_attention_order,
            conv_1d=args.conv_1d,
            padding_mode=args.padding_mode,
            padding=args.padding,
            use_attention=True,
            use_qna=True,
            kernel_size=3,
        )

    else:
        model = MDM(**get_model_args(args, data))
    diffusion = create_gaussian_diffusion(args)
    return model, diffusion


def get_unet_model_args(args, data, num_joints):
    # default args
    clip_version = 'ViT-B/32'
    action_emb = 'tensor'
    cond_mode = 'text'
    if data is not None and hasattr(data, 'dataset') and  hasattr(data.dataset, 'num_actions'):
        num_actions = data.dataset.num_actions
    else:
        num_actions = 1

    # SMPL defaults
    data_rep = 'rot6d'
    njoints = 25
    nfeats = 6

    if args.dataset == 'humanml':
        data_rep = 'hml_vec'
        njoints = 263
        nfeats = 1
    elif args.dataset == 'mixamo':
        data_rep = 'mixamo_vec'
        if data is not None:
            njoints = data.raw_motion.shape[1]
        elif num_joints is not None:
            njoints = num_joints
        else:
            njoints = int(args.num_joints) if args.num_joints else 174
        nfeats = 1

    elif args.dataset == 'bvh_general':
        data_rep = 'bvh_general_vec'
        if data is not None:
            njoints = data.raw_motion.shape[1]
        elif num_joints is not None:
            njoints = num_joints
        else:
            assert args.num_joints
            njoints = int(args.num_joints)
        nfeats = 9 if args.repr == '6d' else 7
        njoints = njoints * nfeats
        nfeats = 1

    return {'modeltype': '', 'njoints': njoints, 'nfeats': nfeats, 'num_actions': num_actions,
            'translation': True, 'pose_rep': 'rot6d', 'glob': True, 'glob_rot': True,
            'latent_dim': 512, 'ff_size': 1024, 'num_layers': 8, 'num_heads': 4,
            'dropout': 0.0, 'activation': "gelu", 'data_rep': data_rep, 'cond_mode': cond_mode,
            'cond_mask_prob': 0.1, 'action_emb': action_emb, 'arch': "trans_enc",
            'emb_trans_dec': False, 'clip_version': clip_version, 'dataset': args.dataset}

def get_model_args(args, data):

    # default args
    clip_version = 'ViT-B/32'
    action_emb = 'tensor'
    cond_mode = get_cond_mode(args)
    if hasattr(data.dataset, 'num_actions'):
        num_actions = data.dataset.num_actions
    else:
        num_actions = 1

    # SMPL defaults
    data_rep = 'rot6d'
    njoints = 25
    nfeats = 6

    if args.dataset == 'humanml':
        data_rep = 'hml_vec'
        njoints = 263
        nfeats = 1
    elif args.dataset == 'kit':
        data_rep = 'hml_vec'
        njoints = 251
        nfeats = 1

    return {'modeltype': '', 'njoints': njoints, 'nfeats': nfeats, 'num_actions': num_actions,
            'translation': True, 'pose_rep': 'rot6d', 'glob': True, 'glob_rot': True,
            'latent_dim': 512, 'ff_size': 1024, 'num_layers': 8, 'num_heads': 4,
            'dropout': 0.0, 'activation': "gelu", 'data_rep': "hml_vec", 'cond_mode': "text",
            'cond_mask_prob': 0.1, 'action_emb': None, 'arch': args.arch,
            'emb_trans_dec': False, 'clip_version': clip_version, 'dataset': args.dataset}


def create_gaussian_diffusion(args):
    # default params
    predict_xstart = True  # we always predict x_start (a.k.a. x0), that's our deal!
    steps = args.diffusion_steps
    scale_beta = 1.  # no scaling
    timestep_respacing = ''  # can be used for ddim sampling, we don't use it.
    learn_sigma = False
    rescale_timesteps = False

    betas = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_beta)
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not args.sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        lambda_vel=args.lambda_vel,
        lambda_rcxyz=args.lambda_rcxyz,
        lambda_fc=args.lambda_fc,
    )

def create_motion_unet(
    motion_args,
    image_size,
    num_channels,
    num_res_blocks,
    channel_mult="",
    learn_sigma=False,
    class_cond=False,
    use_checkpoint=False,
    attention_resolutions="16",
    num_heads=1,
    num_head_channels=-1,
    num_heads_upsample=-1,
    use_scale_shift_norm=False,
    dropout=0,
    resblock_updown=False,
    use_fp16=False,
    use_new_attention_order=False,
    conv_1d=False,
    padding_mode='zeros',
    padding=1,
    use_attention=False,
    use_qna=False,
    kernel_size=3,
):
    if channel_mult == "":
        if image_size == 512:
            channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    # if motion_args['dataset'] in ['humanml', 'mixamo', 'bvh_general']:
    #     ch = motion_args['njoints'] * motion_args['nfeats']
    # else:
    #     raise 'dataset is not supported yet.'

    if motion_args['dataset'] == 'humanml':
        data_rep = 'hml_vec'
        motion_args['njoints'] = 263
        motion_args['nfeats'] = 1
    elif motion_args['dataset'] == 'kit':
        data_rep = 'hml_vec'
        motion_args['njoints'] = 251
        motion_args['nfeats'] = 1
        
    ch = motion_args['njoints'] * motion_args['nfeats']


    
    return MDM_UNetModel(
        motion_args=motion_args,
        image_size=image_size,
        in_channels=ch,
        model_channels=num_channels,
        out_channels=ch,
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        dims=1 if conv_1d else 2,
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_new_attention_order=use_new_attention_order,
        padding_mode=padding_mode,
        padding=padding,
        use_attention=use_attention,
        use_qna=use_qna,
        kernel_size=kernel_size,
        clip_version=motion_args['clip_version']
    )