from base import BaseExperiment
from fullpose import FullPose7D
import KITTI
import VIPER
from torchvision import transforms
from torch.utils.data import DataLoader
from model import FullPose7DModel, FullPose7DModelNOLSTM
import torch


class FullSequenceTest(FullPose7D, BaseExperiment):

    @staticmethod
    def submit_arguments(parser):
        parser.add_argument('--max_size', type=int, nargs=3, default=[0, 0, 0])
        parser.add_argument('--sequence', type=int, default=10)
        parser.add_argument('--hidden', type=int, default=500)
        parser.add_argument('--layers', type=int, default=3)
        parser.add_argument('--dataset', type=str, default='KITTI', choices=['KITTI', 'VIPER'])
        parser.add_argument('--sequence_name', type=str, default='001')

    def __init__(self, in_folder, out_folder, args):
        BaseExperiment.__init__(self, in_folder, out_folder, args)

        # Determine size of input images
        _, (tmp, _, _) = next(enumerate(self.testset))
        self.input_size = (tmp.size(3), tmp.size(4))

        # Model
        self.model = FullPose7DModelNOLSTM(
            self.input_size,
            hidden=args.hidden,
            nlayers=args.layers,
        )

        self.model.cuda()
        self.beta = 1

    def load_dataset(self, args):
        # Image pre-processing
        transform = transforms.Compose([
            transforms.Scale(320),
            transforms.CenterCrop((320, 448)),
            transforms.ToTensor(),
        ])

        if args.dataset == 'KITTI':
            self.dataset = KITTI

            test_set = KITTI.Subsequence(
                sequence_length=args.sequence,
                overlap=0,
                transform=transform,
                sequence_numbers=[int(args.sequence_name)],
                relative_pose=False,
            )

        elif args.dataset == 'VIPER':
            self.dataset = VIPER

            val_set = VIPER.Subsequence(
                folder=VIPER.FOLDERS['val'],
                sequence_length=args.sequence,
                overlap=0,
                transform=transform,
                max_size=args.max_size[1],
                sequence_name=args.sequence_name,
                relative_pose=False,
            )

            # Ground truth not available for test folder
            test_set = val_set
        else:
            raise RuntimeError('Unkown dataset: {}'.format(args.dataset))


        dataloader_test = DataLoader(
            test_set,
            batch_size=1,
            pin_memory=False,
            shuffle=False,
            num_workers=args.workers,
            drop_last=True
        )

        return None, None, dataloader_test

    def test(self, dataloader=None):
        dataloader = self.testset

        self.model.eval()
        state = None
        all_outputs = []
        all_targets = []
        all_filenames = []
        prev_image = None
        for i, (images, poses, filenames) in enumerate(dataloader):
            poses.squeeze_(0)
            all_targets.append(poses)

            # Prepend last image from previous sequence
            if prev_image is not None:
                images = torch.cat((prev_image, images), 1)
            prev_image = images[:, -1, :].unsqueeze(0)

            input = self.to_variable(images, volatile=True)
            output, state = self.model(input, state)

            all_outputs.append(output.data)
            all_filenames.extend(filenames)

        all_outputs = torch.cat(all_outputs)
        #all_outputs = torch.cat((torch.zeros(1, 6).cuda(), all_outputs))
        all_targets = torch.cat(all_targets)

        all_outputs = self.convert_pose_to_global(all_outputs.cpu())

        self.visualize_paths(all_outputs.unsqueeze(0), all_targets.unsqueeze(0), [all_filenames])


        o = all_outputs.cuda().unsqueeze(0)
        t = all_targets.cuda().unsqueeze(0)

        self.plot_rotation_loss_per_meter(o, t)
        self.plot_translation_loss_per_meter(o, t)


    def restore_from_checkpoint(self, checkpoint):
        self.model.load_state_dict(checkpoint['model'])