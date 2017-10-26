from GTAV_old import read_from_text_file, visualize_predicted_path
from tools.colmap_utils import read_colmap_predictions


if __name__ == '__main__':

    colmap_txt_file = r'C:\Users\Adrian\Desktop\testsequence2\images.txt'
    gta_txt_file = r'C:\Users\Adrian\Desktop\testsequence2\poses.txt'

    colmap_times, colmap_poses = read_colmap_predictions(colmap_txt_file)
    gta_times, gta_poses = read_from_text_file(gta_txt_file)

    colmap_times = list(colmap_times)
    gta_times = list(gta_times)

    # Get only the poses with the corresponding time stamp
    indices = [gta_times.index(t) for t in colmap_times]
    gta_poses = gta_poses[indices, :]

    visualize_predicted_path(colmap_poses, gta_poses, './plot.svg', show_rot=False)