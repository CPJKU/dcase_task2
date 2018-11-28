
import os
import time
import argparse
from dcase_task2.lasagne_wrapper.utils import BColors


cols = BColors()

# define model to train
MODELS_DATA_DICT = dict()
MODELS_DATA_DICT['vgg_gap_spec1_1'] = 'specs_train_v1'
MODELS_DATA_DICT['vgg_gap_spec1_2'] = 'specs_train_v1'
MODELS_DATA_DICT['vgg_gap_spec2'] = 'specs_train_v2'


if __name__ == "__main__":
    """ main """

    # add argument parser
    parser = argparse.ArgumentParser(description='Run entire set of experiments.')
    parser.add_argument('--experiment_set', help='Select set of experiments to run (pretrain, finetune or all)',
                        type=str, default='all')
    parser.add_argument('--num_gpus', help='How many GPUs to spread the experiments over (1..4, while 4 is the default)',
                        type=int, default=4)
    args = parser.parse_args()

    # commands for pre-training the models
    pre_train_commands = []
    for model, data in MODELS_DATA_DICT.iteritems():
        cmd = './run_experiment.sh dcase "python2 train.py --model models/%s.py --data tut18T2-%s --max_len 3000" %d'
        pre_train_commands.append(cmd % (model, data, args.num_gpus))

    # commands for fine-tuning the models with self-verification
    fine_tune_commands = []
    for model, data in MODELS_DATA_DICT.iteritems():
        for i in range(10):

            # self-verification step
            cmd = 'python2 self_verify.py --model models/%s.py --spec_dir %s' % (model, data)
            cmd += ' --top_probs_thresh 0.95 --k_per_class 40 --no_len_fix --min_len 3000 --max_len 3000 --stochastic'
            cmd += ' --train_file train_self_verified_it%d.csv --tag it%d' % (i, i)
            fine_tune_commands.append(cmd)

            # fine-tuning step
            cmd = './run_experiment.sh dcase "python2 train.py --model models/%s.py --data tut18T2ver-%s' % (model, data)
            cmd += ' --ini_params params_foldX_it%d --max_len 3000' % i
            cmd += ' --train_file train_self_verified_it%d.csv --tag it%d' % (i, i + 1)
            cmd += ' --fine_tune"'
            cmd += ' %d' % args.num_gpus
            cmd = cmd.replace("foldX", "%d")
            fine_tune_commands.append(cmd)

    # select experiments to run
    if args.experiment_set == 'pretrain':
        commands = pre_train_commands
    elif args.experiment_set == 'finetune':
        commands = fine_tune_commands
    elif args.experiment_set == 'all':
        commands = pre_train_commands + fine_tune_commands
    else:
        raise ValueError("argument '--set' must one of 'pretrain', 'finetune' or 'all', "
                         "got '%s' instead" % args.experiment_set)

    # run experiments
    cmd_idx = 0
    n_commands = len(commands)
    while cmd_idx < n_commands:
        print(cols.print_colored("Running experiment (%d / %d)" % (cmd_idx + 1, n_commands), cols.WARNING))
        print(commands[cmd_idx])
        os.system(commands[cmd_idx])
        cmd_idx += 1
