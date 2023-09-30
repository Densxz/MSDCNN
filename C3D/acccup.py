f = open('./predict_ret.txt', 'r')
contents = f.readlines()
fl=0
for content in contents:
    value=content.split(',')
    # print(value[0])
    # print(value[2].strip())
    if value[0]!=value[2].strip():
        fl=fl+1
print(len(contents))
print(fl)
print(fl/len(contents))
print(1-fl/len(contents))