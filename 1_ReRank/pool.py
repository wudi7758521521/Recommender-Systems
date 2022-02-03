import multiprocessing
import time
import os

print("温馨提示：本机为", os.cpu_count(), "核CPU")


def func(msg):
    print("msg:", msg)
    # time.sleep(3)
    print("end")


if __name__ == "__main__":
    # 这里开启了4个进程
    pool = multiprocessing.Pool(processes=4)
    for i in range(8):
        msg = "hello %d" % (i)
        pool.apply_async(func, (msg,))

    pool.close()
    pool.join()  # 调用join之前，先调用close函数，否则会出错。执行完close后不会有新的进程加入到pool,join函数等待所有子进程结束
    print("Successfully")
