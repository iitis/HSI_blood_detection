# -*- coding: utf-8 -*-
"""
************************************************************************
Copyright 2020 Institute of Theoretical and Applied Informatics, 
Polish Academy of Sciences (ITAI PAS) https://www.iitis.pl
authors: P. Glomb, pglomb@iitis.pl, M. Romaszewszki, mromaszewski@iitis.pl

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
************************************************************************

Code for experiments in the paper by M. Romaszewski, P.Glomb, M. Cholewa, A. Sochan  
,,A Dataset for Evaluating Blood Detection in Hyperspectral Images''
preprint: arXiv:2008.10254

The set of universal serialization functions.

bz2save(anything, filename) will put anything in file of that name. 
Load it with bz2load(filename). 

bz2save(anything) will return compressed and encoded string.
Load it with bz2load(string).
"""

import bz2
import pickle
from base64 import b64encode, b64decode
import unittest

# ----------------------------------------------------------------------------

def bz2load(fname_or_string):
    try:
        with open(fname_or_string, 'rb') as f:
            data = f.read()
    except IOError:
        data = b64decode(fname_or_string)
    return pickle.loads(bz2.decompress(data))

# ----------------------------------------------------------------------------

def bz2save(data, fname=None):
    compressed = bz2.compress(pickle.dumps(data, pickle.HIGHEST_PROTOCOL))
    if fname is None:
        return b64encode(compressed)
    else:
        with open(fname, 'wb') as f:
            return f.write(compressed)

# ----------------------------------------------------------------------------


class Aaa(object):
    def __init__(self, b=2): self.a = b


class Bz2LoadSaveTest(unittest.TestCase):
       
    def test_1(self, fname='test'):
        import os        

        a = Aaa(42)
        aa = bz2load(bz2save(a))
        self.assertEqual(aa.a, 42)
        bz2save(a, fname)
        aa = bz2load(fname)
        self.assertEqual(aa.a, 42)
        os.remove(fname)
    
    def test_2(self, fname='test'):
        import numpy as np
        import os
        
        b = np.random.randn(11, 12, 13)
        bb = bz2load(bz2save(b))
        np.testing.assert_array_equal(bb, b)
        bz2save(b, fname)
        bb = bz2load(fname)
        np.testing.assert_array_equal(bb, b)
        os.remove(fname)        
    
# ----------------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()