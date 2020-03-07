"""
After moving all the files using the 1_ file, we run this one to extract
the images from the videos and also create a data file we can use
for training and testing later.
"""
import csv
import glob
import os
import os.path
from subprocess import call

def extract_files():
	"""
	Extracting can be done with ffmpeg:
	`ffmpeg -i video.mpg image-%04d.jpg`
	"""
	os.chdir(os.path.abspath('../data'))
	data_file = [['type','class','img_path']]
	folders = ['train', 'val']

	for folder in folders:
		class_folders = glob.glob(os.path.join(folder, '*'))

		for vid_class in class_folders:
			class_files = glob.glob(os.path.join(vid_class, '*.mp4'))

			for video_path in class_files:
				# Get the parts of the file.
				video_parts = get_video_parts(video_path)

				train_or_test, classname, filename_no_ext, filename = video_parts

				# Only extract if we haven't done it yet. Otherwise, just get
				# the info.
				if not check_already_extracted(video_parts):
					# Now extract it.
					src = os.path.join(train_or_test, classname, filename)
					dest = os.path.join(train_or_test, classname,
					                    filename_no_ext + '-%04d.jpg')
					call(["ffmpeg", "-i", src, dest])

				# Now get how many frames it is.
				nb_frames = get_nb_frames_for_video(video_parts)

				# data_file.append([train_or_test, classname, filename_no_ext, nb_frames])

				print("Generated %d frames for %s" % (nb_frames, filename_no_ext))
			img_path = glob.glob(os.path.join(vid_class, '*.jpg'))
			for x in img_path:
				data_file.append([train_or_test, classname, x])
			with open('data_file.csv', 'a+') as fout:
				writer = csv.writer(fout)
				writer.writerows(data_file)
			data_file = []

	print("Extracted and wrote %d video files." % (len(data_file)))

def get_nb_frames_for_video(video_parts):
	"""Given video parts of an (assumed) already extracted video, return
	the number of frames that were extracted."""
	train_or_test, classname, filename_no_ext, _ = video_parts
	generated_files = glob.glob(os.path.join(train_or_test, classname,
	                                         filename_no_ext + '*.jpg'))
	return len(generated_files)

def get_video_parts(video_path):
	"""Given a full path to a video, return its parts."""
	parts = video_path.split(os.path.sep)
	filename = parts[2]
	filename_no_ext = filename.split('.')[0]
	classname = parts[1]
	train_or_test = parts[0]

	return train_or_test, classname, filename_no_ext, filename

def check_already_extracted(video_parts):
	"""Check to see if we created the -0001 frame of this file."""
	train_or_test, classname, filename_no_ext, _ = video_parts
	return bool(os.path.exists(os.path.join(train_or_test, classname,
	                                        filename_no_ext + '-0001.jpg')))

def main():
	extract_files()

if __name__ == '__main__':
	main()
