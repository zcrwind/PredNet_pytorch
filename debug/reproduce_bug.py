# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch.autograd import Variable

# a = Variable(torch.from_numpy(np.arange(12).reshape(3, 4)).float(), requires_grad=True)
L = [Variable(torch.from_numpy(np.arange(12).reshape(3, 4)).float(), requires_grad=True),
	 Variable(torch.from_numpy(np.arange(1, 13).reshape(3, 4)).float(), requires_grad=True),
	 Variable(torch.from_numpy(np.arange(2, 14).reshape(3, 4)).float(), requires_grad=True),
	 Variable(torch.from_numpy(np.arange(3, 15).reshape(3, 4)).float(), requires_grad=True),
	 ]
w = Variable(torch.from_numpy(np.array([1., 0.1, 0.1, 0.1])).float())
# L = [a * i for i in range(1, 4)]
error_list = [i * w for i in L]
error_list = [e.sum() for e in L]
'''
# print(L)
>>> print(L)
[Variable containing:
  0   1   2   3
  4   5   6   7
  8   9  10  11
[torch.LongTensor of size 3x4]
, Variable containing:
  0   2   4   6
  8  10  12  14
 16  18  20  22
[torch.LongTensor of size 3x4]
, Variable containing:
  0   3   6   9
 12  15  18  21
 24  27  30  33
[torch.LongTensor of size 3x4]
]

>>> error_list
[Variable containing:
 66
[torch.LongTensor of size 1]
, Variable containing:
 132
[torch.LongTensor of size 1]
, Variable containing:
 198
[torch.LongTensor of size 1]
]

'''
total = error_list[0]
for e in error_list[1:]:
	total = total + e

total.backward()

