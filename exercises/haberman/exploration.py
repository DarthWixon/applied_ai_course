from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

HABERMAN_FILE_PATH = Path("../datasets/haberman/haberman.csv")

df = pd.read_csv(HABERMAN_FILE_PATH, header=None)

# have a look to check it's loaded
print(df.head())

# rename columns in line with documentation
df.columns = ["age", "op_year", "num_axil_nodes", "surv_status"]

# remap surv_status s.t. 0 = dead, 1=alive
df["surv_status"] = df["surv_status"].replace(to_replace={2.0: 0, 1.0: 1})

# plot histograms of each
dist_plots, ((dist_ax_1, dist_ax_2), (dist_ax_3, dist_ax_4)) = plt.subplots(
    nrows=2, ncols=2
)

dist_ax_1.hist(df["age"])

dist_ax_2.hist(df["op_year"])
dist_ax_3.hist(df["num_axil_nodes"])
dist_ax_4.hist(df["surv_status"])
# plt.show()
plt.close()

# plot pairplot using seaborn
## hue by surv_status to see differences between alive and dead
sns.pairplot(data=df, corner=True, hue="surv_status")
plt.show()
