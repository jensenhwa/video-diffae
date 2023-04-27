from templates import *
from templates_latent import *
from templates_video import video_autoenc

if __name__ == '__main__':
    # train the autoenc moodel
    # this can be run on 2080Ti's.
    print("ONE")
    gpus = [4,5,6,7]
    conf = video_autoenc()
    conf.name = "opticalflow_fixed"
    train(conf, gpus=gpus)

