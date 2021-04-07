

def read_G(file_len):
    GD = []
    GP = []

    f_P = open('./电导数据/' + str(file_len) + '权重GP.txt', 'r')
    f_D = open('./电导数据/'+ str(file_len) + '权重GD.txt', 'r')
    for each_P in f_P:
        GP.append(float(each_P.replace('\n', '')))
    for each_D in f_D:
        GD.append(float(each_D.replace('\n', '')))
    return GP, GD, file_len

input_num = input('输入电导数据的个数：')
GP, GD, G_num = read_G(int(input_num))
print(GP)

