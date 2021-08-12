
from ptype import Column
from ptype.Column import (
    ANOMALIES_INDEX,
    MISSING_INDEX,
    TYPE_INDEX,
    Column,
    _get_unique_vals,
)

import datetime


def quick_timestamp():
    now = datetime.datetime.now()
    print ("Cell Run At: ",now.strftime("%Y-%m-%d %H:%M:%S"))
    return now


def test_function():
    for i in range(0,3):
        print('Hi! my name is,')
    print('Chika chika slim shady.')
