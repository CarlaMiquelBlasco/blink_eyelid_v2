import os
import shutil


def organize_images_and_write_txt(image_folder, eye_pos_file, batch_size):
    '''
    :param image_folder: path containing extracted frames from a video
    :param eye_pos_file: contains eye position for each face detected in new frames and sorted by image index.
    :param batch_size: number of frames taken as context for LSTM
    :return: Reorganize the images into folder of batch_size images
    '''
    output_txt_file = '../Data/test/check/gt_blink.txt'
    output_dir = os.path.dirname(output_txt_file)
    os.makedirs(output_dir, exist_ok=True)
    # Read eye positions
    with open(eye_pos_file, 'r') as f:
        eye_positions = f.readlines()

    # Prepare output .txt file
    with open(output_txt_file, 'w') as out_txt:
        group_count = 1
        batch = []
        eye_pos_batch = []

        for i, line in enumerate(eye_positions):
            image_num = i + 1  # Image numbers start from 1
            image_name = f'{image_num}.bmp'
            image_src_path = os.path.join(image_folder, f'{image_num}face.bmp')

            # Prepare the new path for the image
            folder_path = f'check/blink/{group_count}/{batch_size}'
            image_dst_folder = os.path.join(image_folder, folder_path)
            os.makedirs(image_dst_folder, exist_ok=True)

            # New destination image path
            new_image_path = os.path.join(image_dst_folder, f'{image_num}face.bmp')
            shutil.copy(image_src_path, new_image_path)  # Copy image to the new directory

            # Add image path to batch
            batch.append(f'{folder_path}/{image_name}')
            eye_pos_batch.append(line)

            # Once batch has batch_size images, write them to the output .txt and split the eye_pos_relative.txt
            if len(batch) == batch_size:
                # Write paths to grouped_paths.txt
                out_txt.write(f'1 {" ".join(batch)}\n')

                # Write eye positions to the corresponding folder
                eye_pos_output_path = os.path.join(image_dst_folder, 'eye_pos_relative.txt')
                with open(eye_pos_output_path, 'w') as eye_pos_out:
                    eye_pos_out.writelines(eye_pos_batch)

                # Clear the batch and increment group
                batch = []
                eye_pos_batch = []
                group_count += 1

        # Handle any remaining images (if total number of images is not a multiple of 10)
        if batch:
            out_txt.write(f'1 {" ".join(batch)}\n')
            eye_pos_output_path = os.path.join(image_dst_folder, 'eye_pos_relative.txt')
            with open(eye_pos_output_path, 'w') as eye_pos_out:
                eye_pos_out.writelines(eye_pos_batch)


def create_additional_files(output_folder):
    '''
    :return: Creates additional files for the required input structure os the model.
    '''
    # Ensure directories are created
    os.makedirs('../Data/test/check/unblink', exist_ok=True)

    # List of files to be created
    dirs = [
        '../Data/test/check/gt_blink_left.txt',
        '../Data/test/check/gt_blink_right.txt',
        '../Data/test/check/gt_non_blink.txt',
        '../Data/test/check/gt_non_blink_left.txt',
        '../Data/test/check/gt_non_blink_right.txt'
    ]

    # Ensure all directories for the files exist
    for dir in dirs:
        output_dir = os.path.dirname(dir)
        os.makedirs(output_dir, exist_ok=True)

    # Correct the destination paths in shutil.copy by adding '../' to match the folder structure
    shutil.copy('../Data/test/check/gt_blink.txt', '../Data/test/check/gt_blink_left.txt')
    shutil.copy('../Data/test/check/gt_blink.txt', '../Data/test/check/gt_blink_right.txt')

    # Create empty non-blink files
    with open('../Data/test/check/gt_non_blink.txt', 'w') as file:
        pass
    with open('../Data/test/check/gt_non_blink_right.txt', 'w') as file:
        pass
    with open('../Data/test/check/gt_non_blink_left.txt', 'w') as file:
        pass

    shutil.move(os.path.join(output_folder,'check/blink'), '../Data/test/check/blink')

    shutil.rmtree(os.path.join(output_folder,'check'))


