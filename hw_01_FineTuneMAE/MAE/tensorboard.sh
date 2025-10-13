tensorboard --logdir logs/cifar10

# ssh -L 6006:localhost:6006 name@IP
# http://localhost:6006/





# tensorboard --logdir logs/cifar10/mae-pretrain & # default 6006
# tensorboard --logdir logs/cifar10/pretrain-cls --port 6007 &
# tensorboard --logdir logs/cifar10/scratch-cls/ --port 6008 &
# ssh -L 6006:localhost:6006 -L 6007:localhost:6007 -L 6008:localhost:6008 name@IP
# http://localhost:6006/
# http://localhost:6007/
# http://localhost:6008/


