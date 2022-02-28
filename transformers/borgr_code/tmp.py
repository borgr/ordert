# import json
# import os
#
# root, dirs, filenames = next(os.walk("/cs/labs/daphna/guy.hacohen/borgr/data/openwebtext"))
# length = 1000000
# out_path = f"/cs/labs/daphna/guy.hacohen/borgr/data/openwtxt_{lengtd}.txt"
# print("Writing to out_path")
# with open(out_path, "w")as out_fl:
#     for i, filename in enumerate(filenames):
#         if i % 1000 == 0:
#             print(f"Dealt with {i} files")
#         if i == length:
#             print("Done")
#         with open(filename) as fl:
#             out_fl.write(fl.read())

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import seaborn as sns

sns.set()

df = pd.DataFrame(np.array([[1, 2, 3, 4, 5, 6],[0.1, 0.20, 0.30, 0.40, 0.50, 0.60], [100, 1000, 10000, 100000, 100000, 100000]]).T,
                  columns=["scale", "first", "second"])
# Initialize figure and ax
fig, ax = plt.subplots()
sns.lineplot(x="first", y="scale",
             data=df[df["scale"] < 3])
sns.lineplot(x="second", y="scale",
             data=df)
ax.set(xscale="log")
plt.savefig(os.path.join(".", f"tmp.png"))
plt.clf()
# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
#
# x = 10 ** np.arange(1, 10)
# y = x * 2
# data = pd.DataFrame(data={'x': x, 'y': y})
#
# f, ax = plt.subplots(figsize=(7, 7))
# ax.set(xscale="log", yscale="log")
# sns.regplot("x", "y", data, ax=ax, scatter_kws={"s": 100})
# plt.show()