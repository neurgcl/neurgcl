import unittest
import numpy as np

def bbox_sample(size=None, min=None, max=None):
    use_min_max = not (max is None)
    use_size = not (size is None)
    if (use_min_max and not use_size):
        if min is None:
            min = np.zeros_like(max)
        min=np.array(min)
        max=np.array(max)
        return np.random.uniform(min, max)
    elif (use_size and not use_min_max):
        size=np.array(size)
        return np.random.uniform(-size/2., size/2.)
    else:
        raise ValueError('Either min-max or size should be provided')


class TestFun(unittest.TestCase):
    def setUp(self):
        pass

    def test_sample_size(self):
        ret = bbox_sample([1,2,3])
        self.assertEqual(ret.shape, (3,))

    def test_sample_max(self):
        ret = bbox_sample(max=[1,2,3])
        self.assertEqual(ret.shape, (3,))

    def test_sample_min_max(self):
        ret = bbox_sample(min=[-1,-2,-3], max=[1,2,3])
        self.assertEqual(ret.shape, (3,))

    def test_tmp(self):
        l=[bbox_sample(size=[0.4,0.4,0.4]) for _ in range(100)]
        l=np.array(l)
        print("max,min:", l.max(axis=0), l.min(axis=0))

if __name__ == "__main__":
    unittest.main()
