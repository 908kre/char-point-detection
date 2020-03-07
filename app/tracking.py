import os
from mlboard_client import Writer
MLBOARD_URL = os.getenv('MLBOARD_URL', 'http://192.168.10.8:2020')

if __name__ == "__main__":
    w = Writer(
        MLBOARD_URL,
        'test0',
        {
            'p0': 12,
            'p1': {
                'p3': 'fasdfa'
            },
        },
    )
    for i in range(100):
        w.add_scalars({'aaa' : i, "bbb": i * 2})
