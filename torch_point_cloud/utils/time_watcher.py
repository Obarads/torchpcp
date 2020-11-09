import time

class TimeWatcher:
    def __init__(self):
        self.reset()
    def reset(self):
        r"""
        Reset params.
        """
        self.__start_time = 0
        self.__previous_time = 0
    def start(self):
        r"""
        Start timer and return start time.
        This function do time reset when called.
        """
        self.reset()
        now_time = time.time()
        self.__start_time = now_time
        self.__previous_time = now_time
        return now_time
    def lap_and_split(self):
        r"""
        Return lap time and split time.
        """
        now_time = time.time()
        split_time = now_time - self.__start_time
        lap_time = now_time - self.__previous_time
        self.__previous_time = now_time
        return lap_time, split_time

def timecheck(start=None, publisher="time"):
    if start is not None:
        print("{}: {}s".format(publisher, time.time()-start))
    return time.time()

