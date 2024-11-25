import random
indices = list(range(6))
random.shuffle(indices)  # 随机打乱索引以创建随机配对
pprint.pprint(indices)
pairs = [(indices[i], indices[i + 1]) for i in range(0, len(indices), 2)]
pprint.pprint(pairs)