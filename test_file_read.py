import os
import pandas as pd

# def get_data():
#     path = os.getcwd()+'\\data'
#     name = os.listdir(path)
#     name0 = []
#     l = []
#     for i in name:
#         if i in ['ant','jedit','ivy','synapse','xalan']:
#             new_path = path+'\\'+i
#             output = []
#             for j in os.listdir(new_path):
#                 data = pd.read_csv(new_path+'\\'+j,engine='python')
#                 output.append(data)
#             output = pd.concat(output,axis=0)
#             name0.append(i)
#             l.append(output)
#     return name0,l

# path = os.getcwd()
# name = os.listdir(path+'\\Data')
# for i in name:
#     subpath = path +'\\Data' + '\\' + i
#     subname = os.listdir(subpath)
#     for j in subname:
#         finalpath = subpath + '\\' + j
#         x = pd.read_csv(finalpath,engine='python').dropna(how='all')
#         newpath = path+'\\data'+'\\'+i+'\\'+j
#         print(newpath)
#         x.to_csv(newpath)