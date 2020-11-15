import os

from torch_point_cloud.utils.setting import make_folders

def download_and_unzip(www, output_path):
    zip_file = os.path.basename(www)
    if not os.path.exists(zip_file):
        os.system('wget %s --no-check-certificate' % (www))
    folder_name = zip_file[:-4]
    make_folders(folder_name)
    os.system("unzip %s -d %s" % ('"'+zip_file+'"', "'"+folder_name+"'"))
    os.system('mv %s %s' % ('"'+folder_name+'"', '"'+output_path+'"'))
    os.system('rm %s' % ('"'+zip_file+'"'))


    