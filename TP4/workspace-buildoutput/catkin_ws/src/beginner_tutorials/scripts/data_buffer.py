import math
from collections import deque
import threading


def _compute_distance(data_tuple, stamp):
    return abs(stamp.nsecs - data_tuple[0].nsecs)

def _closest(tuple1, tuple2, stamp):
    """
    Return the data which has the clossest time to stamp
    """
    distance1 = _compute_distance(tuple1, stamp)
    distance2 = _compute_distance(tuple2, stamp)
    if distance1 < distance2:
        return tuple1[1]
    else:
        return tuple2[1]

def _is_greater_than(stamp1, stamp2):
    """
    Return stamp1 >= stamps2
    """
    if stamp1.secs >= stamp2.secs:
        return True
    else:
        return stamp1.nsecs >= stamp2.nsecs


class DataBuffer(deque):

    def __init__(self, *args, **kwargs):
        super(DataBuffer, self).__init__(*args, **kwargs)
        self.buffer_semaphore = threading.Semaphore()

    def append(self, stamp, data):
        self.buffer_semaphore.acquire()
        super(DataBuffer, self).append((stamp, data))
        self.buffer_semaphore.release()

    def get_closest_to(self, stamp):
        """
        Return the stored object wich is the closest to "stamp"
        """
        previous_tuple = (-1, None)
        self.buffer_semaphore.acquire()
        for data_tuple in self:
            if _is_greater_than(data_tuple[0], stamp):
                if previous_tuple[0] == -1:
                    # The first data is bigger than the stamp so the buffer
                    # is probably too small
                    self.buffer_semaphore.release()
                    raise BufferError("{} requested but {} found.".format(stamp, data_tuple[0]))
                res = _closest(previous_tuple, data_tuple, stamp)
                self.buffer_semaphore.release()
                return res
            previous_tuple = data_tuple
        # We didn't found a data that is close enough to the stamp so it
        # means that the requested data has not been processed yet
        self.buffer_semaphore.release()
        raise BufferError("No matched data found in the buffer.")
