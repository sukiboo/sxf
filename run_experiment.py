
import argparse
import yaml
import dill as pickle

from experiment_component.experiment import Experiment
from experiment_component.data_report import DataReport


def load_exp(exp_name):
    '''load experiment data from the given directory'''
    print(f'loading experiment \'{exp_name}\'...')
    try:
        params_exp = yaml.safe_load(open(f'./exp_data/{exp_name}/params.yml'))
        params_exp.update({'exp_name': exp_name})
        exp = Experiment(params_exp)
        with open(f'./exp_data/{exp_name}/data.pkl', 'rb') as save_file:
            exp.__dict__.update({**pickle.load(save_file), 'exp_name': exp_name})
        for agent in exp.agents:
            agent.setup_manager(exp_name)
    except:
        raise NameError(f'\ncannot load experiment {exp_name}...')
    return exp


if __name__ == '__main__':

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', '-l', default=False)
    parser.add_argument('--config', '-c', default='config.yml')
    args = parser.parse_args()

    # load or run the experiment
    if args.load:
        exp = load_exp(args.load)
    else:
        params_exp = yaml.safe_load(open(f'./configs/{args.config}'))
        exp = Experiment(params_exp)
        exp.run()

    # report result of the experiment
    data = DataReport(exp)
    if not args.load:
        data.report()
