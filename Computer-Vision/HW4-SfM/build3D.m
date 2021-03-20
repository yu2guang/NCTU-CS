clear all;

file = '../CV2020_HW4/results1/Mesona/';  % 'Mesona', 'Statue'
src_path = '../CV2020_HW4/data/';
% file = '../CV2020_HW4/our_results2/deer/';
% src_path = '../CV2020_HW4/our_data/';
    

pts3D = csvread([file 'pts3D.csv']);
pts2D = csvread([file 'pts2D.csv']);
CameraMatrix = csvread([file 'CameraMatrix.csv']);

obj_main(pts3D, pts2D, CameraMatrix, [src_path 'Mesona1.JPG'], 1) % 'Mesona1.JPG', 'Statue1.bmp', 'deer1.jpg'