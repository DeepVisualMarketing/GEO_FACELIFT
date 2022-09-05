class Config(object):
    env = 'default'
    backbone = 'resnet50'
    classify = 'softmax'
    num_classes = 2
    metric = 'arc_margin'
    easy_margin = False
    use_se = False
    loss = 'focal_loss'

    display = False
    finetune = False

    train_root = '/home/tech/Workspace/Data/Project_tmp/Facelift/evaluator_arcface'
    val_list = '/data/Datasets/webface/val_data_13938.txt'

    test_root = '/data1/Datasets/anti-spoofing/test/data_align_256'
    test_list = 'test.txt'

    lfw_root = '/data/Datasets/lfw/lfw-align-128'
    lfw_test_list = '/data/Datasets/lfw/lfw_test_pair.txt'

    checkpoints_path = '/home/tech/Workspace/Data/Project_tmp/Facelift/evaluator_arcface/rst/210416'
    save_interval = 2

    train_batch_size = 16  # batch size
    test_batch_size = 60

    input_shape = (3, 224, 224)

    optimizer = 'sgd'

    use_gpu = True  # use GPU or not
    gpu_id = '0, 1'
    num_workers = 4  # how many workers for loading data
    print_freq = 1000  # print info every N batch

    debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'

    max_epoch = 100
    lr = 1e-4 #1e-4  # initial learning rate # increase lr from epoch 310
    lr_step = 10
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 5e-6 #5e-4
