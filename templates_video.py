from templates import ffhq64_autoenc, ffhq128_autoenc_130M


def shanghai_64_autoenc():
    conf = ffhq64_autoenc()
    conf.data_name = 'shanghailmdb'
    conf.eval_num_images = 5_000
    conf.eval_every_samples = 500_000
    conf.eval_ema_every_samples = 500_000
    conf.total_samples = 10_000_000
    conf.name = 'shanghai_autoenc'
    return conf


def video_64_autoenc():
    conf = ffhq64_autoenc()
    # conf.net_ch_mult = (1, 2, 4, 4)
    conf.net_ch_mult = (1, 1)
    conf.net_enc_channel_mult = (1, 2, 4, 8, 8)
    conf.data_name = 'video'
    conf.eval_num_images = 5_000
    conf.eval_every_samples = 10_000_000
    conf.eval_ema_every_samples = 10_000_000
    conf.total_samples = 1_000_000
    conf.batch_size = 16
    conf.sample_size = 16
    conf.net_ch = 4
    conf.net_beatgans_embed_channels = 512  # size of semantic encoding
    conf.name = 'temp'  # 'video_st_flow+gs'
    return conf


def video_128_autoenc():
    conf = ffhq128_autoenc_130M()
    conf.data_name = 'video'
    conf.net_beatgans_embed_channels = 512
    conf.batch_size = 16
    conf.sample_size = 16
    return conf
