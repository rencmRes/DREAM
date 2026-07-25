import argparse
import os

from torch.utils.data import DataLoader

from data.select_dataset import define_Dataset
from models.select_model import define_Model
from utils import utils_image as util
from utils import utils_option as option


def main(json_path='options/test_DREAM_sr_x2_gan.json'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path)
    args = parser.parse_args()

    opt = option.parse(args.opt, is_train=False)
    opt = option.dict_to_nonedict(opt)

    test_set = define_Dataset(opt['datasets']['test'])
    test_loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )

    model = define_Model(opt)
    model.load()

    for test_data in test_loader:
        image_name = os.path.basename(test_data['L_path'][0])
        image_stem = os.path.splitext(image_name)[0]

        model.feed_data(test_data)
        model.test()
        output = util.tensor2uint(model.current_visuals()['E'])

        save_path = os.path.join('result', '{}.tif'.format(image_stem))
        util.imsave(output, save_path)
        print(save_path)


if __name__ == '__main__':
    main()
