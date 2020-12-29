from typing import List
import numpy as np

def read(file_path:str):
    """
    Get the point cloud from the pcd file.

    Parameter
    ---------
    file_path : str
        PCD file path.

    Returns
    -------
    point_cloud : np.ndarray
        point cloud
    file_format_str
        file format
    """
    with open(file_path, "rb") as f:
        # Get binary str.
        binary_str = f.read()

        # Convert binary str to common str.
        common_str = binary_str.decode()

        # Convert str to list.
        list_str = common_str.split("\n")

        # Remove comment-only lines and empty row.
        list_str_wo_comments = [x.split(" ") for x in list_str if len(x) is not 0 if x[0] is not "#"]
        # The above is replaced by:
        # list_str_wo_comments = []
        # for x in list_str:
        #     if len(x) is not 0: # Remove empty row.
        #         if x[0] is not "#": # Remove comment-only lines
        #             list_str_wo_comments.append(x)

        # The next row of 'DATA' is a point cloud.
        point_cloud_str = list_str_wo_comments[10:]

        # convert str to np.float32
        point_cloud = np.array(point_cloud_str, dtype=np.float32)

        # Get the format of this file.
        file_format_str = list_str_wo_comments[:10]

        return point_cloud, file_format_str

def write(file_path:str, )