import os
import cv2
import glob
import traceback

def create_mjpeg(image_folder, filename_pattern,  output_filename, fps=2):
    
    images = glob.glob(image_folder + filename_pattern) 
    images.sort() 
    print("images {}".format(images))
    if not images:
        print("Not found image files")
        return

    frame = cv2.imread(images[0])
    height, width, layers = frame.shape

    video = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height))

    for image in images:
        frame = cv2.imread(image)
        video.write(frame)

    cv2.destroyAllWindows()
    video.release()
    print("Created mjpg file {}".format(output_filename))
    
 
if __name__ == "__main__":
  try:
    images_dir = "./epoch_change_infer/"
    image_files = glob.glob("./mini_test/images/*.*")
    image_files.sort()
    basename = os.path.basename(image_files[0])
    name = basename.split(".")[0]
    
    filename_pattern = "Epoch*_" + name + ".jpg"
    print("--- filename_pattern {}".format(filename_pattern))
    output_dir = "./inference-video"
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
    output_filename = os.path.join(output_dir, name + ".mjpeg")
    
    create_mjpeg(images_dir, filename_pattern,  output_filename, fps=1)
    
  except:
    traceback.print_exc()
    

