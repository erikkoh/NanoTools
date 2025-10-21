import os
def All_tem_paths(path):
    dm3_list = []
    mib_list = []
    extra_list = []
    for (root,dirs,files) in os.walk(path):
        for file in files:
            if file.split(".")[-1] == "dm3":
                dm3_list.append(os.path.join(root,file))
            elif file.split(".")[-1] == "mib":
                mib_list.append(os.path.join(root,file))
    return dm3_list, mib_list

