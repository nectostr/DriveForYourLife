import requests as re
import numpy as np
import matplotlib.pyplot as plt
import json
import config
import time
import os
import warnings
warnings.filterwarnings("ignore")
import numba

times = []
total_global_privat_pic_num_special_for_timur = 0


def way_foe_loosers(way):

    for i in way:
        print(f"({i//width}, {i%width})", end=" ")
    print()

def check_if_we_moved(car_id="0"):
    url = f'https://localhost:8080/babbage/api/v1/world'
    r = re.post(url, verify=False).json()
    return r["cars"][car_id]["position"]

def create_beaute():
    global total_global_privat_pic_num_special_for_timur
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
    # for i in r["cars"]:
    # with "0" as i:
    if True:
        i = '0'
        cars = r["cars"][i]["position"]
        arr[cars // height, cars % width, 1] = 200
    plt.imshow(arr)
    plt.savefig(f"data/{total_global_privat_pic_num_special_for_timur}.jpg")
    total_global_privat_pic_num_special_for_timur += 1

def bfs(matrix, w, h, start, all_appropriete):
    to_see = []
    visited = []
    to_see.append([start])
    while to_see:# and len(to_see[-1]) < 15: #if way to long
        closesed = min([abs(start-i) for i in all_users_origin])
        to_see.sort(reverse=False, key=lambda x:
        abs(x[-1] - closesed))
        # min([(x[-1]//h - i//h)**2+(x[-1]%w - i%w)**2
        #      for i in all_users_origin] if len(to_see[0])<10 else
        #     lambda x: 1-len(x)))
        path = to_see[-1]
        curr = path[-1]
        visited.append(curr)
        del to_see[-1]
        if curr not in all_appropriete:
            # if len(to_see) < 50:
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


# @numba.jit()
def build_c_m(field):
    height, width = field.shape
    conf_matrix = np.zeros((width * height, width * height), dtype=int)
    for fi_x in range(height):
        for fi_y in range(width):
            i = fi_x*height + fi_y
            for mx, my in ((-1, 0), (1, 0), (0, 1), (0, -1)):
                if not mx == my and field[fi_x, fi_y] != 0:
                    if 0 <= fi_x + mx < width and 0 <= fi_y + my < height:
                        j = (fi_x + mx)* height + (fi_y + my)
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
    headers = {'content-type': 'application/json', "Authorization": config.game_token
    }
    url = f'https://localhost:8080/{config.token}/api/v1/actions'
    data = {"Type": "move", "Action":
            {
            "CarId": car_id,
            "MoveDirection": move
            }
            }
    r = re.post(url, data=json.dumps(data), verify=False, headers=headers)
    # print(r)




if __name__ == '__main__':
    # TODO check if person exist
    for i in os.listdir("data"):
        os.remove("data\\"+i)
    # fig = plt.gcf()
    # fig.show()
    # fig.canvas.draw()
    create_beaute()
    url = f'https://localhost:8080/babbage/api/v1/world'
    r = re.post(url, verify=False).json()
    width = r["width"]
    height = r["height"]
    start_mask_side = 9
    print(f"mask_size = {start_mask_side}")
    field = np.zeros((width, height), dtype=int)
    for i in range(len(r["grid"])):
        if r["grid"][i]:
            field[i // height, i % width] = 1
    a_mat = build_c_m(field)  # f(field)
    all_users_origin = [r["customers"][i]["origin"] for i in r['customers']]
    start_to_id = {r["customers"][i]["origin"]: i for i in r['customers']}

    # for i in r["cars"]:
    i = "0"

    car_start = r["cars"][i]["position"]
    car_capacity = 1#r["cars"][i]["capacity"]
    curr_capacity = 0
    way = []
    all_important_ahead = []
    way += bfs(a_mat, width, height, car_start, all_users_origin)
    if not way:
        print("Not found closest point")
        time.sleep(100)
    all_important_ahead.append(way[-1])
    way_to_fin = bfs(a_mat, width, height, way[-1], [r["customers"][start_to_id[way[-1]]]["destination"]])

    if way_to_fin:
        way += way_to_fin
    else:
        print("THis is fake")
        way += [way[-1]-1, way[-1]-2, way[-1]-3] #TODO shiiiiit - if rock?
    all_important_ahead.append(way[-1])
    # TODO check if correct points
    start_mask = build_start_mask()
    fin_mask = build_fin_mask()
    prev = car_start
    move = 0
    real_prev, real_curr = car_start, car_start
    while way:
        create_beaute()
        t1 = time.time()
        curr = way[0]
        if curr > prev:
            if prev + 1 == curr:
                move = 1
            else:
                move = 0
        else:
            if prev - 1 == curr:
                move = 3
            else:
                move = 2
        move_car(car_id=int(i), move=move)
        real_curr = check_if_we_moved()
        if real_curr != curr:
            del way[0]
        else:
            print("Do_not_move", move)
            continue
        # TODO rebuild all users_origins

        # TODO REbuild, not from the start finMask via move
        start_mask = build_start_mask()
        fin_mask = build_fin_mask()
        if not curr in all_important_ahead:
            if curr_capacity <= car_capacity:

                possible = set(all_users_origin) - (
                        set(all_users_origin) - set(start_mask))
                # print(f"way before is {way}")

                for point in possible:
                    if car_capacity <= curr_capacity:
                        break
                    curr_capacity+=1
                    rebuild_list = list(set(all_important_ahead\
                                   + list(possible) + \
                                   [r["customers"][start_to_id[i]]["destination"]
                                    for i in possible]))
                    new_way = []
                    while rebuild_list:
                        new_way += bfs(a_mat, width, height, curr, rebuild_list)
                        if new_way:
                            del rebuild_list[rebuild_list.index(new_way[-1])]

                    way = new_way
        else:
            del all_important_ahead[all_important_ahead.index(curr)]
            if curr not in all_users_origin:
                curr_capacity -= 1
        prev = curr
        way_foe_loosers(way)
        # print(time.time() - t1)
        if time.time() - t1 < 0.2:
            print("waited")
            time.sleep(time.time() - t1)

time.sleep(100)



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