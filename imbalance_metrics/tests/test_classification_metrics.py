import unittest
from imbalance_metrics import classification_metrics as cm
import get_y
import numpy as np


class TestCM(unittest.TestCase):

    def setUp(self):
        self.y,self.y_pred,self.y_proba=get_y.cla()

    def test_get_minority(self):
        self.assertEqual(cm.get_minority(y_true=self.y), 0)

    def test_gmean_score(self):
        self.assertEqual(cm.gmean_score(y_true=self.y, y_pred=self.y_pred), 0.18989464186862232)

    def test_pr_davis(self):
        self.assertEqual(cm.pr_davis(self.y,self.y_proba), 0.528087977617617)

    def test_pr_manning(self):
        self.assertEqual(cm.pr_manning(self.y,self.y_proba), 0.546410265625786)



if __name__ == '__main__':
    unittest.main()