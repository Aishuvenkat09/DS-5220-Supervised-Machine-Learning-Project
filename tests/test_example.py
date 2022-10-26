import numpy as np
from utils import create_array

def test_equal_array():
  arr1= create_array()
  arr2= creare_array()
  assert (np.array_equal(arr1, arr2)), "Arrays are not equal"
