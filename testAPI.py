import requests as re
import numpy as np
import matplotlib.pyplot as plt
import json
import config
import time
import numba

times = []

def dec(f):
    def w(*args, **kwargs):
        global times
        t1 = time.time()
        r = f(*args, **kwargs)
        times.append(time.time() - t1)
        return r
    return w

@dec
def f(field):
    return build_c_m(field)

@numba.jit()
def build_c_m(field):
    height, width = field.shape
    conf_matrix = np.zeros((width * height, width * height), dtype=int)
    for fi_x in range(height):
        for fi_y in range(width):
            i = fi_x * fi_y
            for mx, my in ((-1, 0), (1, 0), (0, 1), (0, -1)):
                if not mx == my and field[fi_x, fi_y] != 0:
                    if 0 <= fi_x + mx < width and 0 <= fi_y + my < height:
                        j = (fi_x + mx) * (fi_y + my)
                        if field[fi_x + mx, fi_y + my] != 0:
                            conf_matrix[i, j] = 1





url = f'https://localhost:8080/babbage/api/v1/world'
r = re.post(url, verify=False).json()
width = r["width"]
height = r["height"]
print(height, width)

fig = plt.gcf()
fig.show()
fig.canvas.draw()

for run in range(1000):
    try:
        url = f'https://localhost:8080/babbage/api/v1/world'
        r = re.post(url, verify=False).json()
        arr = np.zeros((width, height, 3), dtype=int)
        field = np.zeros((width, height), dtype=int)
        conf_matrix = np.zeros((width * height, width*height), dtype=int)
        for i in range(len(r["grid"])):
            if r["grid"][i]:
                arr[i // height, i % width, 0] = 50
                arr[i // height, i % width, 1] = 50
                arr[i // height, i % width, 2] = 50
                field[i // height, i % width] = 1


        for i in r["customers"]:
            # if int(i) < int(max(r["customers"]))//2:
            user_starts = r["customers"][i]["origin"]
            arr[user_starts // height, user_starts % width,0] = 100
            # arr[user_starts // r["height"], user_starts % r["width"],1] = 0
            # arr[user_starts // r["height"], user_starts % r["width"],2] = 0
            # if int(i) > int(max(r["customers"]))//2:
            user_to = r["customers"][i]["destination"]
            arr[user_to // height, user_to % width, 2] = 100
        for i in r["cars"]:
            cars = r["cars"][i]["position"]
            arr[cars // height, cars % width, 1] = 100

        conf_matrix = f(field)

        plt.imshow(arr)
        fig.canvas.draw()
    except:
        break

print(times)
print(np.mean(times))



"""
data:
- now, 
- future points list
- points to rebuild list

alg:
build matrix
get closer

for step (check if future pers still there for all future points, resize mask )
    
    search thoose who in current mask and end in total mask
        (- person reachable (start and finsh))
        (- took all) / (- took one)
    if yes - rebuild all points
    no - counter + (if enlarge mask (bases on capasity))
    build route me-he_start-he_end (+ mask) кратчайший остовной граф
"""