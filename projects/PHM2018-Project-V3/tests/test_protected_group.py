import unittest
import pandas as pd
from src.protected_group import protected_mask,load

class ProtectedGroupTests(unittest.TestCase):
    def test_future_guard_before_holdout_is_strict(self):
        bounds=pd.DataFrame([dict(scheme='temporal',split='holdout',Tool='01M01',start=10000)])
        m=pd.DataFrame({'Tool':['01M01']*4,'time':[8199,8200,9999,10000]})
        self.assertEqual(protected_mask(m,bounds).tolist(),[True,False,False,False])

    def test_protected_group_has_no_holdout_role(self):
        with self.assertRaises(ValueError):load('holdout','A','maximum')

if __name__=='__main__':unittest.main()
