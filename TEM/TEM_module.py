import os
import hyperspy.api as hs
import pyxem as pxm
import matplotlib.patheffects as patheffects
from zarr import ZipStore
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

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

def Make_file_dict(dm3_list, mib_list):
    files = {}
    name = ""
    for dm3 in dm3_list:
        name = dm3.split('/')[-1].split('.')[0]
        for mib in mib_list:
            if name == mib.split('/')[-1].split('.')[0]:
                files.update({name : (dm3, mib)})
    return files


def get_folders(directory):
    import os
    paths = []
    for root, dirs, files in os.walk(directory):
        for dir in dirs:
            paths.append(os.path.join(root, dir))
    return paths



def convert(dm3_list, mib_list):
    
    #Create folder for saving
    dst_folder = "zspy_files"
    os.makedirs(dst_folder, exist_ok=True)

    #Create list of paths to files wanted to convert
    file_dict = Make_file_dict(dm3_list, mib_list)

    #Load and convert files
    for file in file_dict.values():
        #print(file[0])

        #Load dm3 file
        s_adf = hs.load(file[0])

        #Find ref values for loading mib file correctly
        x_akse_adf = s_adf.axes_manager[0]
        y_akse_adf = s_adf.axes_manager[1]

        #Load and crop mib file
        s_4d = hs.load(file[1], lazy=True, navigation_shape=(x_akse_adf.size+2, y_akse_adf.size))
        s_4d_crop = s_4d.inav[:-2]

        #Label axes correctly
        s_4d_crop.axes_manager[0].scale = x_akse_adf.scale
        s_4d_crop.axes_manager[1].scale = y_akse_adf.scale
        s_4d_crop.axes_manager[0].units = x_akse_adf.units
        s_4d_crop.axes_manager[1].units = x_akse_adf.units

        #Store files
        filename = file[0].split('/')[-1].split('.')[0] + ".zspy"
        store  = ZipStore(dst_folder + '/' + filename)

        #Save file in correct folder
        s_4d_crop.save(store, chunks=(64,64,64,64))
        #print("File " + filename + "converted to .zspy")


def magnetic_process(path):

    #Create folder for saving
    dst_folder = "magnetic_figures"
    os.makedirs(dst_folder, exist_ok=True)

    #Convert all zspy files
    for (root,dirs,files) in os.walk(path):
        for file in files:
            new_path = dst_folder + '/' + file + ".png"
            #print(new_path)
            #print(os.path.join(root,file))
            try:
                #Load file
                s = hs.load(os.path.join(root, file), lazy=True)

                #Create and set naviagor
                s_sum = s.sum(axis=(-1,-2))
                s_nav = s_sum.transpose()
                s_nav.compute()
                s.navigator = s_nav

                #Find beam signal, and correct with plane
                s_bs = s.get_direct_beam_position(method="center_of_mass")
                s_bs.compute()
                s_bs_lp = s_bs.get_linear_plane()
                s_bs_corr = s_bs - s_bs_lp

                #Plot
                fig,ax = plt.subplots()
                pxm.utils.plotting.plot_beam_shift_color(s_bs_corr, ax = ax, ax_indicator = ax.inset_axes([0.75, 0.05, 0.2, 0.2]))

                #Add scalebar to figure
                scalebar_kwargs = {'size' : 2, 'label' :  "2 um", 'loc' : 3, 'frameon' : False, 'color' : 'white', 'size_vertical': 0.2}
                scalebar = AnchoredSizeBar(ax.transData, **scalebar_kwargs)
                scalebar.txt_label._text.set_path_effects([patheffects.withStroke(linewidth=2, foreground='black', capstyle="round")])
                ax.add_artist(scalebar)

                #Save figure in dst folder
                plt.savefig(new_path)
                
                print(f"converted image: {file}")
            except:
                print(f"failed to load file: {file}")
    
    