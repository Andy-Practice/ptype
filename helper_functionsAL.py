import datetime


def quick_timestamp():
    now = datetime.datetime.now()
    print ("Cell Run At: ",now.strftime("%Y-%m-%d %H:%M:%S"))
    return now


def test_function():
    for i in range(0,2):
        print('Hi! my name is,')
    print('Chika chika slim shady.')
