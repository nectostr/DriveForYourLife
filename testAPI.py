import requests as re
import numpy as np
import matplotlib.pyplot as plt
import json
import config
import time
import numba

times = []

def create_beaute():
    url = f'https://localhost:8080/babbage/api/v1/world'
    r = re.post(url, verify=False).json()
    width, height = r["width"], r["height"]
    arr = np.zeros((width, height, 3), dtype=int)
    for i in range(len(r["grid"])):
        if r["grid"][i]:
            arr[i // height, i % width, 0] = 50
            arr[i // height, i % width, 1] = 50
            arr[i // height, i % width, 2] = 50

    for i in r["customers"]:
        user_starts = r["customers"][i]["origin"]
        arr[user_starts // height, user_starts % width, 0] = 100
        user_to = r["customers"][i]["destination"]
        arr[user_to // height, user_to % width, 2] = 100
    for i in r["cars"]:
        cars = r["cars"][i]["position"]
        arr[cars // height, cars % width, 1] = 100
    plt.imshow(arr)
    fig.canvas.draw()

def bfs(matrix, w, h, start, all_appropriete):
    to_see = []
    visited = []
    to_see.append([start])
    while to_see:
        #TODO sort to see via h???
        path = to_see[-1]
        curr = path[-1]
        visited.append(curr)
        del to_see[-1]
        if curr not in all_appropriete:
            for i in np.argwhere(matrix[curr]>0):
                if i[0] not in visited:
                    to_see.insert(0, path + list(i))
        else:
            return path[1:]
    return []



def dec(f):
    def w(*args, **kwargs):
        global times
        t1 = time.time()
        r = f(*args, **kwargs)
        times.append(time.time() - t1)
        return r
    return w


#@numba.jit()
def build_c_m(field):
    height, width = field.shape
    conf_matrix = np.zeros((width * height, width * height), dtype=int)
    for fi_x in range(height):
        for fi_y in range(width):
            i = fi_x + fi_y * height
            for mx, my in ((-1, 0), (1, 0), (0, 1), (0, -1)):
                if not mx == my and field[fi_x, fi_y] != 0:
                    if 0 <= fi_x + mx < width and 0 <= fi_y + my < height:
                        j = (fi_x + mx) + (fi_y + my) * height
                        if field[fi_x + mx, fi_y + my] != 0:
                            conf_matrix[i, j] = 1
    return conf_matrix

def build_fin_mask():
    return [i*height + j for k in range(0, len(way), start_mask_side // 2)
     for i in range(way[k] // height - start_mask_side // 2,
                    way[k] // height + start_mask_side // 2)
     for j in range(way[k] % width - start_mask_side // 2,
                    way[k] // height + start_mask_side // 2)]


def build_start_mask():
    return [i*height + j for i in range(car_start // height - start_mask_side // 2,
                          car_start // height + start_mask_side // 2)
            for j in range(car_start % width - start_mask_side // 2,
                    car_start // height + start_mask_side // 2)]

def move_car(car_id, move):
    url = f'https://api.citysimulation.eu/babbge/api/v1/actions'
    data = {"Type": "move", "Action":
            {
            "CarId": car_id,
            "MoveDirection": move
            } }
    r = re.post(url, data=data, verify=False)




if __name__ == '__main__':
    # t1 = time.time()
    fig = plt.gcf()
    fig.show()
    fig.canvas.draw()
    create_beaute()
    url = f'https://localhost:8080/babbage/api/v1/world'
    r = re.post(url, verify=False).json()
    width = r["width"]
    height = r["height"]
    start_mask_side = width//10
    field = np.zeros((width, height), dtype=int)
    for i in range(len(r["grid"])):
        if r["grid"][i]:
            field[i // height, i % width] = 1
    a_mat = build_c_m(field)#f(field)
    all_users_origin = [r["customers"][i]["origin"] for i in r['customers']]
    start_to_id = {r["customers"][i]["origin"]: i for i in r['customers']}
    # for i in r["cars"]:
    i = "0"
    create_beaute()
    car_start = r["cars"][i]["position"]
    way = []
    all_important_ahead = []
    way += bfs(a_mat, width, height, car_start, all_users_origin)
    if not way:
        continue
    all_important_ahead.append(way[-1])
    way += bfs(a_mat, width, height, way[-1], [int(start_to_id[way[-1]])])
    all_important_ahead.append(way[-1])
    #TODO check if correct points
    start_mask = build_start_mask()
    fin_mask = build_fin_mask()
    prev = car_start
    move = 0
    while way:
        curr = way[0]
        if curr > prev:
            if prev + 1 == curr:
                move = 1
            else:
                move = 2
        else:
            if prev - 1 == curr:
                move = 3
            else:
                move = 0
        del way[0]
        move_car(car_id=int(i), move=move)
        #TODO rebuild all users_origins

        #TODO REbuild, not from the start finMask via move
        start_mask = build_start_mask()
        fin_mask = build_fin_mask()
        if not curr in all_important_ahead:
            possible = set(all_users_origin) - (
                set(all_users_origin) - set(start_mask))
            for point in possible:
                rebuild_list = all_important_ahead + list(possible)
                new_way = []
                while rebuild_list:
                    new_way += bfs(a_mat, width, height, curr, rebuild_list)
                    del rebuild_list[rebuild_list.index(new_way[-1])]
                way = new_way
        else:
            del all_important_ahead[all_important_ahead.index(curr)]
        prev = curr
        print(curr)





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