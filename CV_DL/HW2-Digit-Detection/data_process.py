import h5py, glob, cv2
from tqdm import tqdm
import pandas as pd


def get_bbox_info(src_path, mat_file='digitStruct.mat'):

    # output training csv
    hdf5_data = h5py.File(src_path + mat_file, 'r')
    img_names = hdf5_data['/digitStruct/name']

    img_df = pd.DataFrame([], columns=['img_name', 'num_digit', 'label', 'bbox_info'])

    for i in tqdm(range(hdf5_data['/digitStruct/bbox'].shape[0])):
        # img name
        img_name = ''.join([chr(v[0]) for v in hdf5_data[img_names[i][0]].value])
        img = cv2.imread(src_path + 'images/' + img_name)
        img_size = '({}, {})'.format(img.shape[1], img.shape[0])  # w, h

        # cv2.rectangle(img, (int(left), int(top)), (int(left+width), int(top+height)), (0, 255, 0), 2)
        # cv2.imwrite('output.jpg', img)

        # digit info
        item = hdf5_data['digitStruct']['bbox'][i].item()
        num_digit = hdf5_data[item]['label'].shape[0]

        row_dict = {'label': '', 'bbox_info': ''}
        row_dict['img_name'] = [img_name]
        row_dict['img_size'] = [img_size]
        row_dict['num_digit'] = [num_digit]
        if num_digit == 1:
            row_dict['label'] = [int(hdf5_data[item]['label'].value[0][0])]

            top = hdf5_data[item]['top'].value[0][0]
            left = hdf5_data[item]['left'].value[0][0]
            height = hdf5_data[item]['height'].value[0][0]
            width = hdf5_data[item]['width'].value[0][0]
            row_dict['bbox_info'] = [
                '({}, {}, {}, {})'.format(int(top), int(left), int(width), int(height))]

            img_df = pd.concat([img_df, pd.DataFrame.from_dict(row_dict, orient='columns')])
        else:
            for j in range(num_digit):
                row_dict['label'] += (str(int(hdf5_data[hdf5_data[item]['label'].value[j].item()].value[0][0]))+'&')

                top = hdf5_data[hdf5_data[item]['top'].value[j].item()].value[0][0]
                left = hdf5_data[hdf5_data[item]['left'].value[j].item()].value[0][0]
                height = hdf5_data[hdf5_data[item]['height'].value[j].item()].value[0][0]
                width = hdf5_data[hdf5_data[item]['width'].value[j].item()].value[0][0]
                row_dict['bbox_info'] += ('({}, {}, {}, {})'.format(int(top), int(left), int(width), int(height))+'&')

            row_dict['label'] = [row_dict['label'].rstrip('&')]
            row_dict['bbox_info'] = [row_dict['bbox_info'].rstrip('&')]
            img_df = pd.concat([img_df, pd.DataFrame.from_dict(row_dict, orient='columns')])


    img_df.to_csv(src_path + 'training_info.csv', index=False)
    print('> Training csv saved\n')


def get_annotation_txt(src_path, mode):
    train_df = pd.read_csv(src_path + mode + 'ing_info.csv')

    for _, row in train_df.iterrows():
        fp = open(src_path + 'labels/' + mode + '/' + row['img_name'].split('.')[0] + '.txt', 'w')
        labels = list(map(int, row['label'].split('&')))
        bboxes_info = [tuple(map(int, b_i.strip('()').split(','))) for b_i in row['bbox_info'].split('&')]
        img_size = tuple(map(int, row['img_size'].strip('()').split(',')))
        for l_i, bbi_i in zip(labels, bboxes_info):
            # label_idx x_center y_center width height
            # coordinates scale to [1, 0]
            top, left, width, height = bbi_i
            left_norm = (left+width/2)/img_size[0]
            left_norm = left_norm if left_norm <= 1 else 1
            top_norm = (top+height/2)/img_size[1]
            top_norm = top_norm if top_norm <= 1 else 1
            w_norm = width/img_size[0]
            w_norm = w_norm if w_norm <= 1 else 1
            h_norm = height/img_size[1]
            h_norm = h_norm if h_norm <= 1 else 1
            if l_i == 10:
                fp.write('0 {} {} {} {}\n'.format(left_norm, top_norm, w_norm, h_norm))
            else:
                fp.write('{} {} {} {} {}\n'.format(l_i, left_norm, top_norm, w_norm, h_norm))
        fp.close()

    print('> annotation.txt saved\n')


if __name__ == '__main__':
    src_path = './data/'
    get_bbox_info(src_path)
    get_annotation_txt(src_path, 'train')