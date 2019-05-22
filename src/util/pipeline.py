import os
import sys
import argparse
import datetime
import json
import zipfile

import torch
from tensorboardX import SummaryWriter


class BasePipeline:
    def __init__(self):
        parser = argparse.ArgumentParser()
        self.add_arguments(parser)
        self.args = parser.parse_args()

        run_name = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S') + (
            '' if len(self.args.run_name) == 0 else '_' + self.args.run_name)

        save_folder = os.path.join('..', 'save', run_name)

        self.run_name = run_name
        self.save_folder = save_folder

        self.writer = SummaryWriter(log_dir=os.path.join(save_folder, 'log'))

    def add_arguments(self, parser):
        parser.add_argument('-run_name', default='', type=str)

    @staticmethod
    def zip_source_code(folder, target_zip_file):
        f_zip = zipfile.ZipFile(target_zip_file, 'w', zipfile.ZIP_STORED)
        for root, dirs, files in os.walk(folder):
            for file in files:
                if file.endswith('.py'):
                    f_zip.write(os.path.join(root, file))
        f_zip.close()

    def run(self):
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

        print('run:', self.run_name)
        print('save folder:', self.save_folder)
        print('args:', json.dumps(self.args.__dict__, indent=4))

        with open(os.path.join(self.save_folder, 'args'), 'w') as f:
            f.write('cwd: ' + os.getcwd() + '\n')
            f.write('cmd: ' + ' '.join(sys.argv) + '\n')
            f.write('args: ' + str(self.args) + '\n')
        self.zip_source_code(folder='.', target_zip_file=os.path.join(self.save_folder, 'src.zip'))


    def save_model(self, save_path, state_dict):
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        torch.save(state_dict, save_path)
        print('model saved at {}'.format(save_path))

    def load_model(self, save_path):
        state_dict = torch.load(save_path)
        print('loaded model at {}'.format(save_path))
        return state_dict

