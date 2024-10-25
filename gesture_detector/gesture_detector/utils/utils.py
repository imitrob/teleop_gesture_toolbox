import sys, os, yaml, inspect, itertools, collections
import numpy as np
from collections import OrderedDict, Counter
from scipy.signal import argrelextrema
from numpy.linalg import LinAlgError

def ros_enabled():
    try:
        from gesture_msgs.msg import Frame as Framemsg
        return True
    except:
        return False

# function for loading dict from file ordered
def ordered_load(stream, Loader=yaml.SafeLoader, object_pairs_hook=OrderedDict):
    class OrderedLoader(Loader):
        pass
    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))
    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping)
    return yaml.load(stream, OrderedLoader)

class cc:
    H = '\033[95m'
    OK = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    W = '\033[93m'
    F = '\033[91m'
    E = '\033[0m'
    B = '\033[1m'
    U = '\033[4m'

def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]


class CustomDeque(collections.deque):
    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)

    def __getitem__(self, slic):
        if len(self) == 0: return None
        if isinstance(slic, slice):
            start, stop, step = slic.start, slic.stop, slic.step
            #print(f"start {start}")
            if isinstance(start, int) and start < 0: start = len(self)+start
            if isinstance(stop, int) and stop < 0: stop = len(self)+stop
            #print(f"start {start}")
            if start is not None: start = np.clip(start, 0, len(self)-1)
            if stop is not None: stop = np.clip(stop, 0, len(self)-1)
            #print(f"start {start} stop {stop}")
            return list(itertools.islice(self, start, stop, step))
            return collections.deque(itertools.islice(self, start, stop, step))
        else:
            try:
                return super(CustomDeque, self).__getitem__(slic)
            except IndexError:
                return None
    @property
    def empty(self):
        if len(self) > 0:
            return False
        else:
            return True

    def get_last(self, nlast):
        assert nlast>0
        return self[-nlast:]

    def get_last_with_delay(self, nlast, delay):
        assert delay>=0
        assert nlast>0
        return self[-nlast-delay-1:-delay-1]

    def to_ids(self, seq):
        return seq #[i for i in seq]

    def get_last_common(self, nlast, threshold=0.0):
        last_seq = self.to_ids(self.get_last(nlast))
        if last_seq is None: return None
        most_common = Counter(last_seq).most_common(1)[0]
        if float(most_common[1]) / nlast >= threshold:
            return most_common[0]
        else:
            return None

    def get_last_commons(self, nlast, threshold=0.0, most_common=2, delay=0):
        last_seq = self.to_ids(self.get_last_with_delay(nlast, delay))
        most_commons = Counter(last_seq).most_common(2)
        n = len(last_seq)
        for i in range(len(most_commons)):
            most_commons[i] = (most_commons[i][0], most_commons[i][1]/n)

        ret = []
        for i in range(len(most_commons)):
            #if most_commons[i][1] >= threshold:
            ret.append(most_commons[i][0])
        return ret

'''
class CustomCounter():
    def __init__(self, data):
        self.data = data
    def most_common(self, n, fromkey=1):
        counts = {}
        for d in self.data:
            counts[d[fromkey]] += 1

counts = {'a':3, 'b':2, 'c':1, 'e':3}

for i in range(n):
    m = max(counts)
    counts.pop(m)
'''

class GestureQueue(CustomDeque):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def to_ids(self, seq):
        return [i[1] for i in seq]

def customdeque_tester():
    d = CustomDeque(maxlen=5)
    d.append(1)
    d.append(2)
    d.append(3)
    d.append(4)
    d.append(5)
    d.append(6)
    d
    assert d[0] == 2
    d[-1:]
    assert d[-1] == 6
    d[-1:]
    assert d[-1:] == [6]
    d[-2:]
    assert d[-2:] == [5,6]
    d[:]
    assert d[:] == [2,3,4,5,6]
    d[0:]
    assert d[0:] == [2,3,4,5,6]

    d.get_last(3)

    d.append(1)
    d.append(1)
    d
    assert d.get_last_common(3) == 1
    assert d.get_last_common(3, threshold=0.7) is None
    d
    d.get_last_common(20, threshold=0.0)

    d.append(6)
    d.append(6)
    d.append(1)
    d
    assert d.get_last_common(3, threshold=0.0) == 6
    assert d.get_last_common(5, threshold=0.0) == 1


    d = CustomDeque(maxlen=5)
    assert d[-1] is None
    assert d[0] is None
    assert d[100] is None

    assert d[-10:] is None
    assert d.get_last(1) is None

    d.get_last_common(5, threshold=0.0)

    d = CustomDeque(maxlen=5)
    d.append(1)
    d.append(1)
    d.append(2)
    d.append(2)
    d.append(3)
    d.append(3)
    d.append(3)
    d.append(3)
    d.append(4)
    d.append(5)
    d.append(6)
    d.get_last_commons(20, threshold=0.0)



    d[-20:]

    gq = GestureQueue(maxlen=5)
    gq.append((123,'abc',567))
    gq.append((124,'dcb',678))
    gq.append((124,'abc',678))
    gq.append((124,'abc',678))
    gq.get_last_common(10, threshold=0.0)
