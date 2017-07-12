import unittest
import torch
import dataimport.utils as utils


class TestUtils(unittest.TestCase):

    def test_relative_pose(self):
        print(torch.rand(2,2))
        m1 = torch.FloatTensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])

        # yaw, pitch roll: 90, 45, 45
        m2 = torch.FloatTensor([0, 0.707, -0.707],
                               [-0.707, 0.500, 0.500],
                               [0.707, 0.500, 0.500])

        utils.to_relative_poses([m1, m2])
        print(m1, m2)
        self.assertEqual('foo'.upper(), 'FOO')


if __name__ == '__main__':
    unittest.main()