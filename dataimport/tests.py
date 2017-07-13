import unittest
import torch
import dataimport.utils as utils
import mathutils as math


class TestUtils(unittest.TestCase):

    def test_relative_pose1(self):
        m1 = torch.eye(3, 4)

        # yaw, pitch roll: 90, 45, 45
        m2 = torch.FloatTensor([[0, 0.707, -0.707, 0],
                                [-0.707, 0.500, 0.500, 0],
                                [0.707, 0.500, 0.500, 0]])

        [r1, r2] = utils.to_relative_poses([m1, m2])

        self.assertTrue(torch.equal(r1, torch.eye(3, 4)))
        self.assertTrue(torch.equal(r2, m2))

    def test_relative_pose2(self):
        rot1 = math.rotY(90)
        rot2 = math.rotX(45)
        rot3 = math.rotZ(90)
        rot4 = math.rotY(-90)
        t = torch.zeros(3, 1).double()

        m1 = math.to_affine(torch.mm(rot2, rot1), t)
        m2 = math.to_affine(rot3, t)

        m12 = math.to_affine(torch.mm(rot4, rot2), t)

        [r1, r2] = utils.to_relative_poses([m1, m2])

        print(torch.mm(m12[0:3, 0:3], m1[0:3, 0:3]), m2[0:3, 0:3])

        print(r1, r2)

        self.assertTrue(torch.equal(r1, torch.eye(3, 4).double()))
        self.assertTrue(torch.equal(r2, m12))


if __name__ == '__main__':
    unittest.main()