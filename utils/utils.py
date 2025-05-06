import os.path
import yaml

def get_max_len_128(class_idx):
    dict_maxlen = {'0': [64, (8, 8), (7, 9)],  # (N M), (K, L)
                   '1': [128, (8, 16), (7, 9)],
                   '2': [128, (8, 16), (7, 9)],
                   '3': [128, (8, 16), (7, 9)],
                   '4': [128, (8, 16), (7, 9)],
                   }

    return dict_maxlen[str(class_idx)]


def save_file_setup(args, exp_name):
    if args.benchmark=='TNF':
        log_path = './exp_logs_TNF/{}'.format(exp_name)
        try:
            os.makedirs(log_path, exist_ok=True)  # Creates directory if it doesn't exist
        except FileExistsError:
            pass  # Ignore the error if the directory already exists
        log_txt_path = os.path.join(log_path, '{}-{}-log.txt'.format(args.subscore, args.rater_ID))
        log_curve_path = os.path.join(log_path, '{}-{}-loss.png'.format(args.subscore, args.rater_ID))
        checkpoint_path = os.path.join(log_path, '{}-{}-model.pth'.format(args.subscore, args.rater_ID))
        best_checkpoint_path = os.path.join(log_path, '{}-{}-model_best.pth'.format(args.subscore, args.rater_ID))
    elif args.benchmark == 'NeuroFace':
        log_path = './exp_logs_NeuroFace/{}'.format(exp_name)
        try:
            os.makedirs(log_path, exist_ok=True)  # Creates directory if it doesn't exist
        except FileExistsError:
            pass  # Ignore the error if the directory already exists
        log_txt_path = os.path.join(log_path, 'class-{}-fold-{}-log.txt'.format(args.class_idx, args.fold))
        log_curve_path = os.path.join(log_path, 'class-{}-fold-{}-loss.png'.format(args.class_idx, args.fold))
        checkpoint_path = os.path.join(log_path, 'class-{}-fold-{}-model.pth'.format(args.class_idx, args.fold))
        best_checkpoint_path = os.path.join(log_path, 'class-{}-fold-{}-model_best.pth'.format(args.class_idx, args.fold))
    else:
        log_path = './exp_logs_PD_new/{}'.format(exp_name)
        try:
            os.makedirs(log_path, exist_ok=True)  # Creates directory if it doesn't exist
        except FileExistsError:
            pass  # Ignore the error if the directory already exists
        log_txt_path = os.path.join(log_path, 'class_{}-log.txt'.format(args.class_idx))
        log_curve_path = os.path.join(log_path, 'class_{}-loss.png'.format(args.class_idx))
        checkpoint_path = os.path.join(log_path, 'class_{}-model.pth'.format(args.class_idx))
        best_checkpoint_path = os.path.join(log_path, 'class_{}-model_best.pth'.format(args.class_idx))
    return log_txt_path, log_curve_path, checkpoint_path, best_checkpoint_path


def config_setup(args):
    args.config = 'configs/{}.yaml'.format(args.config_name)
    config = get_config(args)
    merge_config(config, args)



def get_config(args):
    print('Load config yaml from %s' % args.config)
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    return config


def merge_config(config, args):
    for k, v in config.items():
        setattr(args, k, v)


def create_experiment_dir(args):
    if not os.path.exists(args.experiment_path):
        os.makedirs(args.experiment_path)
        print('Create experiment path successfully at %s' % args.experiment_path)


def save_experiment_config(args):
    config_path = os.path.join(args.experiment_path, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(args.__dict__, f)
        print('Save the Config file at %s' % config_path)




