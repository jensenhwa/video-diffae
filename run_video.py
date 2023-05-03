from templates import *
from templates_latent import *
from templates_video import video_autoenc

if __name__ == '__main__':
    # train the autoenc moodel
    # this can be run on 2080Ti's.
    print("ONE")
    gpus = [0,1,2,3]
    conf = video_autoenc()
    conf.name = "temp"
    train(conf, gpus=gpus)

