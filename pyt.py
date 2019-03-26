
#%% [markdown]
# vscode notebook
#  这是一级标题。
#  # 这是一个二级标题。
#  ## 这是一个三级标题
# ### 这是一个四级标题


#%%
msg = '12312'
print(msg)

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))
ts = ts.cumsum()
ts.plot()