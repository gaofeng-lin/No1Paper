#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch

if torch.cuda.device_count() > 1:
    print("我可以使用多gpu训练...........")
else:
    print("no mutile gpu")