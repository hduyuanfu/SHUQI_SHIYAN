import queue
import threading
import time
import random

q_data = queue.Queue(10)
do_thread_num = 5

def getOne(one, j):
    time.sleep(random.random()*3)
    print('线程序号%d,获取元素%d'%(j, one))

class MyThread(threading.Thread):
    def __init__(self, func, data, j):
        threading.Thread.__init__(self)
        self.data = data
        self.j = j
        self.func = func
    def run(self):
        while self.data.qsize()>0:
            self.func(self.data.get(), self.j)
if __name__ == '__main__':
    for data in range(do_thread_num*2):
        q_data.put(data)
    for j in range(do_thread_num-4):
        t1 = MyThread(getOne, q_data, j)
        t1.start()
