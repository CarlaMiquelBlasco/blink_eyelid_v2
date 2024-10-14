import os
import torch
import torch.nn.parallel
import torch.optim
from config import cfg
from networks import atten_net
from networks.blink_eyelid_net import BlinkEyelidNet
from dataloader.HUST_LEBW import HUST_LEBW
import numpy as np
from tqdm import tqdm
import argparse
import csv
from dataloader.preproces_clean import preproces
import shutil
import subprocess


#os.environ["CUDA_VISIBLE_DEVICES"]="2"

def csv_collator(samples):
    sample = samples[0]
    imgs=sample[0]
    eye_poses=sample[1]
    blink_label=[]
    blink_label.append(sample[2])
    for i in range(1,len(samples)):
        sample = samples[i]
        img=sample[0]
        eye_pos=sample[1]
        imgs=torch.cat((imgs,img),0)
        eye_poses=torch.cat((eye_poses,eye_pos),0)
        blink_label.append(sample[2])
    blink_labels=torch.stack(blink_label)
    return imgs,eye_poses,blink_labels

def main(args):
    # Device setup
    cfg.time_size = args.time_size
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("USING MPS")
    else:
        device = torch.device("cpu")
        print("USING CPU")
        # Load the models as before

    # From an input video, generate 25 frames per second and store in '../source_videos/Frames'
    print(os.getcwd())
    video_files = [f for f in os.listdir('../source_videos') if f.endswith('.mp4')]
    # Iterate over each video
    for video_file in video_files:
        video_path = os.path.join('../source_videos', video_file)
        video_name = os.path.splitext(video_file)[0]  # Get video name without extension

        # Create a separate folder for frames of this video
        video_frames_folder = os.path.join('../source_videos/',video_name,'/Frames/Original')
        os.makedirs(video_frames_folder, exist_ok=True)

        # Construct the FFmpeg command for extracting frames
        ffmpeg_command = [
            'ffmpeg',
            '-i', video_path,                   # Input video file
            '-vf', 'fps=25',                    # Set the frames per second
            f'{video_frames_folder}/frame_%08d.bmp'
        ]

        # Run the FFmpeg command
        subprocess.run(ffmpeg_command, check=True)
        print(f"Frames for {video_file} saved in {video_frames_folder}")


        # Preprocess original frames stored in ../source_videos/video_name/Frames/Original:
        preproces(args, video_frames_folder, video_name)
        print(f"Frames preprocessed and reorganized for vide {video_name}.")

        for i in range(2):
            args.eye = 'left' if i == 0 else 'right'
            print(f"Testing for {args.eye} eye")

            atten_generator = atten_net.__dict__[cfg.model](cfg.output_shape, cfg.num_class, cfg)
            atten_generator = torch.nn.DataParallel(atten_generator, device_ids=[0]).to(device)

            blink_eyelid_net = BlinkEyelidNet(cfg).to(device)
            blink_eyelid_net = torch.nn.DataParallel(blink_eyelid_net, device_ids=[0]).to(device)

            cfg.eye = args.eye_type
            test_loader = torch.utils.data.DataLoader(
                HUST_LEBW(cfg, train=False),
                batch_size=1, shuffle=False,
                num_workers=1, pin_memory=True, collate_fn=csv_collator, drop_last=False)

            # Load model checkpoints as before
            checkpoint_file = os.path.join(args.checkpoint_dir, args.eye_type, 'atten_generator.pth.tar')
            checkpoint_file2 = os.path.join(args.checkpoint_dir, args.eye_type, 'blink_eyelid_net.pth.tar')

            atten_generator.load_state_dict(torch.load(checkpoint_file, map_location=device)['state_dict'])
            blink_eyelid_net.load_state_dict(torch.load(checkpoint_file2, map_location=device)['state_dict'])


            print("=> loaded checkpoint '{}'".format(checkpoint_file))

            atten_generator.eval()
            blink_eyelid_net.eval()
            print('testing...')

            start_time = 0
            end_time = 0.04
            with open(os.path.join('../source_videos/',video_name, '/Predictions/predictions_'+cfg.eye+'_'+str(cfg.time_size)+'.csv'), mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['VideoID', 'Frame', 'StartTime','EndTime','BlinkStatus'])  # Write CSV header


                for i, (inputs, pos, blink_label) in enumerate(tqdm(test_loader)):

                    with torch.no_grad():
                        input_var = torch.autograd.Variable(inputs.to(device))

                        global_outputs, refine_output = atten_generator(input_var)  # refineout:(b*t, 2, 64, 48)

                        height = np.int64(0.4*refine_output.shape[2])*4 # height = 100
                        width = height
                        if args.eye_type == 'right':
                            outputs, b = blink_eyelid_net(input_var, height, width, pos.numpy(), torch.chunk(refine_output, 2, 1)[1], device)
                        else:
                            outputs, b = blink_eyelid_net(input_var, height, width, pos.numpy(), torch.chunk(refine_output, 2, 1)[0], device)

                        _, predicted = torch.max(outputs.data, 1)
                        predict=predicted.data.cpu().numpy()
                        #print(f"path {cfg.time_size*i+1}, prediction: {predict}")
                        # Write the same prediction for every image in the folder
                        for j in range(cfg.time_size):
                            writer.writerow([
                                cfg.VideoID,
                                f"{i*cfg.time_size+j}",
                                round(start_time, 2),
                                round(end_time,2),
                                'blink' if predict[0] == 1 else 'no blink'
                            ])
                            start_time += 0.04
                            end_time += 0.04

            # Metrics computed for frames in folder. Optionally, remove the cropped frames so we don't run out of storage for next processings.
            print(f"Metrics computed for {video_name}. Deleting both original and cropped frames in order to increase the available storage")
            #shutil.rmtree(frames_path)
            #shutil.rmtree(os.path.join(frames_path, '../Face_Detected'))

                    #target=blink_label.data.numpy()
                    #for (pre,tar) in zip(predict,target):

                    #    if (abs(tar-1)<1e-5):
                    #      blink_count+=1
                    #      if (abs(pre-1)<1e-5):
                    #          blink_right+=1
                    #    else :
                    #      unblink_count+=1
                    #      if (abs(pre-0)<1e-5):
                    #          unblink_right+=1

            #Recall=blink_right/(blink_count)
            #Precision=blink_right/(blink_right+unblink_count-unblink_right)
            #F1=2.0/(1.0/Recall+1.0/Precision)
            #print(f'{args.eye_type} eye: F1 = {F1}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='configs')
    parser.add_argument('-t', '--time_size', default=10, type=int, metavar='N',
                        help='time size hyperparameter for configuration')

    parser = argparse.ArgumentParser(description='configs')
    parser.add_argument('-e', '--eye_type', default='left', type=str, metavar='N',
                        help='left or right eye (in image)')
    parser.add_argument('-c', '--checkpoint_dir', default='./pretrained_models',type=str, metavar='PATH',
                        help='path to load checkpoint (default: checkpoint)')

    parser.add_argument('-t', '--time_size', default=10, type=int, metavar='N',
                        help='time size hyperparameter for configuration')

    args = parser.parse_args()
    os.chdir('/Users/carlamiquelblasco/Library/Mobile Documents/com~apple~CloudDocs/Desktop/MASTER BERGEN/Q1/NONMANUAL/BLINK_EYELID/blink_eyelid2/256.192.model')

    main(args)
