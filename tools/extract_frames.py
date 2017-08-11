from PIL import Image
import os.path as path
import os
import shutil
import imageio
import argparse
import glob


def extract_frames(videofile, fps=30, new_height=None):
    dir, name_ext = path.split(videofile)
    name, _ = path.splitext(name_ext)
    output_folder = path.join(dir, name)

    if path.isdir(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    video = imageio.get_reader(videofile, 'ffmpeg')
    metadata = video.get_meta_data()
    duration = metadata['duration']
    nframes = metadata['nframes']
    width, height = metadata['source_size']

    new_width = int(width * new_height / height) if new_height else width
    new_height = new_height if new_height else height

    original_fps = nframes / duration
    step = int(original_fps / fps)

    indices = list(range(0, nframes, step))
    timestamps = [int(1000 * duration * (i / nframes)) for i in indices]

    for index, time in zip(indices, timestamps):
        array = video.get_data(index)
        fname = '{}.jpg'.format(time)
        image = Image.fromarray(array)

        #if new_height:
        image = image.resize((new_width, new_height), Image.BICUBIC)
        image.save(path.join(output_folder, fname))


def get_video_files(folder, extension='mp4'):
    return glob.glob(path.join(folder, '*.{}'.format(extension)))


if __name__ == '__main__':
    # Install ffmpeg if not already installed
    imageio.plugins.ffmpeg.download()

    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, default='./',
                        help='The folder that contains the videos.')

    parser.add_argument('--fps', type=int, default=30,
                        help='Number of frames to be extracted per second.')

    parser.add_argument('--height', type=int, default=None,
                        help='Images will be rescaled to the given height.')

    parser.add_argument('--ext', type=str, default='mp4',
                        help='File extension of the videos.')

    args = parser.parse_args()

    files = get_video_files(args.folder, args.ext)
    print('Found {} files in {}'.format(len(files), args.folder))

    for filename in files:
        print('Extracting frames from {}'.format(filename))
        extract_frames(filename, fps=args.fps, new_height=args.height)

    print('Done.')
