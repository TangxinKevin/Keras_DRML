import os
import random
import numpy as np
import glob

AU_name = ['au1', 'au2', 'au4', 'au6', 'au9', 'au12',
    'au15', 'au17', 'au25', 'au26']

def image_normalization(image):
    
    image = image.astype('float32') / 255.
    image = image - np.mean(image)

    return image
    

class ConstructSet():
    def __init__(self, root_dir, num_au_labels, val_ratio):
        self.root_images_dir = os.path.join(root_dir, 'images')
        self.root_labels_dir = os.path.join(root_dir, 'labels') 
        self.val_ratio = val_ratio
        self.num_au_labels = num_au_labels
        
        all_folders = self.load_data_folders(self.root_images_dir)
        train_folders, val_folders = self.train_test_folders(
            all_folders, self.val_ratio)
        self.train = self.read_set(train_folders)
        self.val = self.read_set(val_folders)



    def load_data_folders(self, path):
        
        seq_list = []
        for subdir in sorted(os.listdir(path)):
            if os.path.isdir(os.path.join(path, subdir)):
                seq_list.append(os.path.join(path, subdir))
        
        return seq_list

    def train_test_folders(self, data_list, val_ratio=0.1):
        
        random.shuffle(data_list)
        val_list = data_list[:int(len(data_list) * val_ratio)]
        train_list = data_list[int(len(data_list) * val_ratio):]
        return train_list, val_list

    
    def read_folder_to_image_label(self, image_folder):
        
        dat = []
        label_folder = image_folder.split('/')[-1].split('_')[-1]
        dir_label_folder = os.path.join(self.root_labels_dir, label_folder)
        
        txt_files = [os.path.join(dir_label_folder, label_folder+'_' + i + '.txt') 
            for i in AU_name]

        for i, pt in enumerate(txt_files):
            print(pt)

            with open(pt) as f:
                content = f.readlines()
            if i == 0:
                Frames = len(content)
                contents = np.zeros((Frames, self.num_au_labels))
            content = np.array([x.strip('\n').split(',') for x in content],
                dtype=int)[:, 1]
            content[content > 0] = 1
            contents[:, i] = content

        for fj in range(1, Frames+1):
            if os.path.exists(os.path.join(image_folder, 
                'frameZ'+str(fj)+'.jpg')):
                dat.append([os.path.join(image_folder, 
                    'frameZ'+str(fj)+'.jpg'), list(contents[fj-1, :])])
            else:
                pass
        return dat
    
    def read_set(self, data_set):
        
        data = []
    
        for set_i in data_set:
            dat_per_set = self.read_folder_to_image_label(set_i)

            for dat_item in dat_per_set:
                data.append(dat_item)
        
        return data


def statistics(pred, y, thresh):
    batch_size = pred.shape[0]
    class_nb = pred.shape[1]

    pred = pred > thresh

    statistics_list = []
    for j in range(class_nb):
        TP = 0
        FP = 0
        FN = 0
        TN = 0
        for i in range(batch_size):
            if pred[i][j] == 1:
                if y[i][j] == 1:
                    TP += 1
                elif y[i][j] == 0:
                    FP += 1
                else:
                    assert False
            elif pred[i][j] == 0:
                if y[i][j] == 1:
                    FN += 1
                elif y[i][j] == 0:
                    TN += 1
                else:
                    assert False
            else:
                assert False
        statistics_list.append({'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN})
    return statistics_list

def cal_multi_label_accuracy(y_true, y_pred, threshold):
    y_pred = y_pred > threshold
    accuracy = np.mean(np.equal(y_true, y_pred), axis=0)
    total_accuracy = np.mean(accuracy)
    return total_accuracy, accuracy 


def calc_f1_score(statistics_list):
    f1_score_list = []

    for i in range(len(statistics_list)):
        TP = statistics_list[i]['TP']
        FP = statistics_list[i]['FP']
        FN = statistics_list[i]['FN']

        precise = TP / (TP + FP + 1e-20)
        recall = TP / (TP + FN + 1e-20)
        f1_score = 2 * precise * recall / (precise + recall + 1e-20)
        f1_score_list.append(f1_score)
    mean_f1_score = sum(f1_score_list) / len(f1_score_list)

    return mean_f1_score, f1_score_list

            
if __name__ == '__main__':
    gdata = Gernatate_Data('/home/user/Documents/dataset/DISFA',
        12, 0.2)
    train_set = gdata.train
    val_set = gdata.val

    
   


        


    



    
