import datetime
def random(ids):
    now=datetime.datetime.now()
    q1 = (sum(ids[(now.hour + now.minute) % 4]) * now.second) % 7 + 1
    q2_1=(sum(ids[(now.hour+now.minute)%4])*now.second)%7+1
    q2_2=(sum(ids[(now.hour+now.minute)%4])*now.second)%7+1
    q3_1=(sum(ids[(now.hour+now.minute)%4])*now.second)%7+1
    q3_2=(sum(ids[(now.hour+now.minute)%4])*now.second)%7+1
    q4=(sum(ids[(now.hour+now.minute)%4])*now.second)%7+1
    while q1 > 9:
        q1 = (sum(ids[(now.hour + now.minute) % 4]) * now.second) % 7 + 1
    while q2_1<10 or q2_1>29:
        q2_1 = (sum(ids[(now.hour + now.minute) % 4]) * now.second) % 7 + 10
    while q2_2<10 or q2_2>29:
        q2_2 = (sum(ids[(now.hour + now.minute) % 4]) * now.second) % 7 + 12
    while q3_1<19 or q3_1>30:
        q3_1 = (sum(ids[(now.hour + now.minute) % 4]) * now.second) % 7 + 18
    while q3_2<19 or q3_2>30:
        q3_2 = (sum(ids[(now.hour + now.minute) % 4]) * now.second) % 7 + 19
    while q4<31 or q4>36:
        q4 = (sum(ids[(now.hour + now.minute) % 4]) * now.second) % 7 + 29
    print(q1,q2_1,q2_2,q3_1,q3_2,q4)



ids = [[3, 1, 2, 1, 8, 0, 7, 8, 9], [2, 0, 7, 2, 2, 9, 9, 3, 1], [2, 0, 6, 4, 9, 0, 7, 9, 9],
       [2, 0, 8, 5, 1, 0, 2, 4, 8]]
random(ids)