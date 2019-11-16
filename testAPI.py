import requests as re
import numpy as np
import matplotlib.pyplot as plt
import json
import config
import time
import numba

times = []


def bfs(matrix, w, h, start, all_appropriete):
    to_see = []
    to_see.append([start])
    while to_see:
        path = to_see[-1]
        curr = path[-1]
        del to_see[-1]
        if to_see not in all_appropriete:
            for i in range(w*h):
                if matrix[curr // h, i] == 1:
                    to_see.insert(0, path + [i])
        else:
            return path[1:]
    return False



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
    return conf_matrix




url = f'https://localhost:8080/babbage/api/v1/world'
r = re.post(url, verify=False).json()
width = r["width"]
height = r["height"]
field = np.zeros((width, height), dtype=int)
for i in range(len(r["grid"])):
    if r["grid"][i]:
        field[i // height, i % width] = 1
a_mat = f(field)
all_users = [r["customers"][i]["origin"] for i in r['customers']]
start_id = {r["customers"][i]["origin"]: i for i in r['customers']}
car_start = r["cars"]["0"]["position"]
way = []
all_important_ahead = []
way += bfs(a_mat, width, height, car_start, all_users)
all_important_ahead.append(way[-1])
way += bfs(a_mat, width, height, way[-1], [start_id[way[-1]]])
all_important_ahead.append(way[-1])
while way:
    curr = way[0]
    del way[0]
    #TODO move finMask
    if curr in all_important_ahead:
        pass
    else:
        pass

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
    if reached - nimus this
    else:
    search thoose who in current mask and end in total mask
        (- person reachable (start and finsh))
        (- took all) / (- took one)
    if yes - rebuild all points
    no - counter + (if enlarge mask (bases on capasity))
    build route me-he_start-he_end (+ mask) кратчайший остовной граф
"""