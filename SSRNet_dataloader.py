import os
import torch
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms

# Define a conversion relationship to change image data to Tensor format
dataTransform = transforms.Compose([
    transforms.Resize(64),
    # Convert to Tensor format
    transforms.ToTensor()

])

# Create a new dataset class, inherit the data.Dataset parent class in PyTorch
class AgeDataset(data.Dataset):
    # Default constructor, incoming dataset category, and dataset path
    def __init__(self, mode, dir):
        self.mode = mode
        # Create a new image list, used to store the image path
        self.list_img = []
        # Create a new label list to store the age label
        self.list_label = []
        # Record dataset size
        self.data_size = 0
        # Conversion relationship
        self.transform = dataTransform

        # Extract the image path and label
        if self.mode == 'train':
            dir = dir + '/train/'  # The training set path is in "dir"/train/
            for file in os.listdir(dir):  # Traverse the dir folder
                self.list_img.append(dir + file)  # Add the image path and file name to the image list
                self.data_size += 1  # Data set size increased by 1
                name = file.split(sep='.')  # Split the file name: "1.0.jpg" will be split into "1", ".", "jpg"

                if name[0] == '1':
                    self.list_label.append(1)  # Label is 0 when the picture is a age 1
                elif name[0] == '2':
                    self.list_label.append(2)
                elif name[0] == '3':
                    self.list_label.append(3)
                elif name[0] == '4':
                    self.list_label.append(4)
                elif name[0] == '5':
                    self.list_label.append(5)
                elif name[0] == '6':
                    self.list_label.append(6)
                elif name[0] == '7':
                    self.list_label.append(7)
                elif name[0] == '8':
                    self.list_label.append(8)
                elif name[0] == '9':
                    self.list_label.append(9)
                elif name[0] == '10':
                    self.list_label.append(10)
                elif name[0] == '11':
                    self.list_label.append(11)
                elif name[0] == '12':
                    self.list_label.append(12)
                elif name[0] == '13':
                    self.list_label.append(13)
                elif name[0] == '14':
                    self.list_label.append(14)
                elif name[0] == '15':
                    self.list_label.append(15)
                elif name[0] == '16':
                    self.list_label.append(16)
                elif name[0] == '17':
                    self.list_label.append(17)
                elif name[0] == '18':
                    self.list_label.append(18)
                elif name[0] == '19':
                    self.list_label.append(19)
                elif name[0] == '20':
                    self.list_label.append(20)
                elif name[0] == '21':
                    self.list_label.append(21)
                elif name[0] == '22':
                    self.list_label.append(22)
                elif name[0] == '23':
                    self.list_label.append(23)
                elif name[0] == '24':
                    self.list_label.append(24)
                elif name[0] == '25':
                    self.list_label.append(25)
                elif name[0] == '26':
                    self.list_label.append(26)
                elif name[0] == '27':
                    self.list_label.append(27)
                elif name[0] == '28':
                    self.list_label.append(28)
                elif name[0] == '29':
                    self.list_label.append(29)
                elif name[0] == '30':
                    self.list_label.append(30)
                elif name[0] == '31':
                    self.list_label.append(31)
                elif name[0] == '32':
                    self.list_label.append(32)
                elif name[0] == '33':
                    self.list_label.append(33)
                elif name[0] == '34':
                    self.list_label.append(34)
                elif name[0] == '35':
                    self.list_label.append(35)
                elif name[0] == '36':
                    self.list_label.append(36)
                elif name[0] == '37':
                    self.list_label.append(37)
                elif name[0] == '38':
                    self.list_label.append(38)
                elif name[0] == '39':
                    self.list_label.append(39)
                elif name[0] == '40':
                    self.list_label.append(40)
                elif name[0] == '41':
                    self.list_label.append(41)
                elif name[0] == '42':
                    self.list_label.append(42)
                elif name[0] == '43':
                    self.list_label.append(43)
                elif name[0] == '44':
                    self.list_label.append(44)
                elif name[0] == '45':
                    self.list_label.append(45)
                elif name[0] == '46':
                    self.list_label.append(46)
                elif name[0] == '47':
                    self.list_label.append(47)
                elif name[0] == '48':
                    self.list_label.append(48)
                elif name[0] == '49':
                    self.list_label.append(49)
                elif name[0] == '50':
                    self.list_label.append(50)
                elif name[0] == '51':
                    self.list_label.append(51)
                elif name[0] == '52':
                    self.list_label.append(52)
                elif name[0] == '53':
                    self.list_label.append(53)
                elif name[0] == '54':
                    self.list_label.append(54)
                elif name[0] == '55':
                    self.list_label.append(55)
                elif name[0] == '56':
                    self.list_label.append(56)
                elif name[0] == '57':
                    self.list_label.append(57)
                elif name[0] == '58':
                    self.list_label.append(58)
                elif name[0] == '59':
                    self.list_label.append(59)
                elif name[0] == '60':
                    self.list_label.append(60)
                elif name[0] == '61':
                    self.list_label.append(61)
                elif name[0] == '62':
                    self.list_label.append(62)
                elif name[0] == '63':
                    self.list_label.append(63)
                elif name[0] == '64':
                    self.list_label.append(64)
                elif name[0] == '65':
                    self.list_label.append(65)
                elif name[0] == '66':
                    self.list_label.append(66)
                elif name[0] == '67':
                    self.list_label.append(67)
                elif name[0] == '68':
                    self.list_label.append(68)
                elif name[0] == '69':
                    self.list_label.append(69)
                elif name[0] == '70':
                    self.list_label.append(70)
                elif name[0] == '71':
                    self.list_label.append(71)
                elif name[0] == '72':
                    self.list_label.append(72)
                elif name[0] == '73':
                    self.list_label.append(73)
                elif name[0] == '74':
                    self.list_label.append(74)
                elif name[0] == '75':
                    self.list_label.append(75)
                elif name[0] == '76':
                    self.list_label.append(76)
                elif name[0] == '77':
                    self.list_label.append(77)
                elif name[0] == '78':
                    self.list_label.append(78)
                elif name[0] == '79':
                    self.list_label.append(79)
                elif name[0] == '80':
                    self.list_label.append(80)
                elif name[0] == '81':
                    self.list_label.append(81)
                elif name[0] == '82':
                    self.list_label.append(82)
                elif name[0] == '83':
                    self.list_label.append(83)
                elif name[0] == '84':
                    self.list_label.append(84)
                elif name[0] == '85':
                    self.list_label.append(85)
                elif name[0] == '86':
                    self.list_label.append(86)
                elif name[0] == '87':
                    self.list_label.append(87)
                elif name[0] == '88':
                    self.list_label.append(88)
                elif name[0] == '89':
                    self.list_label.append(89)
                else:
                    self.list_label.append(90)

        elif self.mode == 'validate':
            dir = dir + '/validate/'  # The validate set path is "dir"/test/
            for file in os.listdir(dir):
                self.list_img.append(dir + file)  # Add the path to image list
                self.data_size += 1
                name = file.split(sep='.')

                if name[0] == '1':
                    self.list_label.append(1)  # Label is 0 when the picture is a age 1
                elif name[0] == '2':
                    self.list_label.append(2)
                elif name[0] == '3':
                    self.list_label.append(3)
                elif name[0] == '4':
                    self.list_label.append(4)
                elif name[0] == '5':
                    self.list_label.append(5)
                elif name[0] == '6':
                    self.list_label.append(6)
                elif name[0] == '7':
                    self.list_label.append(7)
                elif name[0] == '8':
                    self.list_label.append(8)
                elif name[0] == '9':
                    self.list_label.append(9)
                elif name[0] == '10':
                    self.list_label.append(10)
                elif name[0] == '11':
                    self.list_label.append(11)
                elif name[0] == '12':
                    self.list_label.append(12)
                elif name[0] == '13':
                    self.list_label.append(13)
                elif name[0] == '14':
                    self.list_label.append(14)
                elif name[0] == '15':
                    self.list_label.append(15)
                elif name[0] == '16':
                    self.list_label.append(16)
                elif name[0] == '17':
                    self.list_label.append(17)
                elif name[0] == '18':
                    self.list_label.append(18)
                elif name[0] == '19':
                    self.list_label.append(19)
                elif name[0] == '20':
                    self.list_label.append(20)
                elif name[0] == '21':
                    self.list_label.append(21)
                elif name[0] == '22':
                    self.list_label.append(22)
                elif name[0] == '23':
                    self.list_label.append(23)
                elif name[0] == '24':
                    self.list_label.append(24)
                elif name[0] == '25':
                    self.list_label.append(25)
                elif name[0] == '26':
                    self.list_label.append(26)
                elif name[0] == '27':
                    self.list_label.append(27)
                elif name[0] == '28':
                    self.list_label.append(28)
                elif name[0] == '29':
                    self.list_label.append(29)
                elif name[0] == '30':
                    self.list_label.append(30)
                elif name[0] == '31':
                    self.list_label.append(31)
                elif name[0] == '32':
                    self.list_label.append(32)
                elif name[0] == '33':
                    self.list_label.append(33)
                elif name[0] == '34':
                    self.list_label.append(34)
                elif name[0] == '35':
                    self.list_label.append(35)
                elif name[0] == '36':
                    self.list_label.append(36)
                elif name[0] == '37':
                    self.list_label.append(37)
                elif name[0] == '38':
                    self.list_label.append(38)
                elif name[0] == '39':
                    self.list_label.append(39)
                elif name[0] == '40':
                    self.list_label.append(40)
                elif name[0] == '41':
                    self.list_label.append(41)
                elif name[0] == '42':
                    self.list_label.append(42)
                elif name[0] == '43':
                    self.list_label.append(43)
                elif name[0] == '44':
                    self.list_label.append(44)
                elif name[0] == '45':
                    self.list_label.append(45)
                elif name[0] == '46':
                    self.list_label.append(46)
                elif name[0] == '47':
                    self.list_label.append(47)
                elif name[0] == '48':
                    self.list_label.append(48)
                elif name[0] == '49':
                    self.list_label.append(49)
                elif name[0] == '50':
                    self.list_label.append(50)
                elif name[0] == '51':
                    self.list_label.append(51)
                elif name[0] == '52':
                    self.list_label.append(52)
                elif name[0] == '53':
                    self.list_label.append(53)
                elif name[0] == '54':
                    self.list_label.append(54)
                elif name[0] == '55':
                    self.list_label.append(55)
                elif name[0] == '56':
                    self.list_label.append(56)
                elif name[0] == '57':
                    self.list_label.append(57)
                elif name[0] == '58':
                    self.list_label.append(58)
                elif name[0] == '59':
                    self.list_label.append(59)
                elif name[0] == '60':
                    self.list_label.append(60)
                elif name[0] == '61':
                    self.list_label.append(61)
                elif name[0] == '62':
                    self.list_label.append(62)
                elif name[0] == '63':
                    self.list_label.append(63)
                elif name[0] == '64':
                    self.list_label.append(64)
                elif name[0] == '65':
                    self.list_label.append(65)
                elif name[0] == '66':
                    self.list_label.append(66)
                elif name[0] == '67':
                    self.list_label.append(67)
                elif name[0] == '68':
                    self.list_label.append(68)
                elif name[0] == '69':
                    self.list_label.append(69)
                elif name[0] == '70':
                    self.list_label.append(70)
                elif name[0] == '71':
                    self.list_label.append(71)
                elif name[0] == '72':
                    self.list_label.append(72)
                elif name[0] == '73':
                    self.list_label.append(73)
                elif name[0] == '74':
                    self.list_label.append(74)
                elif name[0] == '75':
                    self.list_label.append(75)
                elif name[0] == '76':
                    self.list_label.append(76)
                elif name[0] == '77':
                    self.list_label.append(77)
                elif name[0] == '78':
                    self.list_label.append(78)
                elif name[0] == '79':
                    self.list_label.append(79)
                elif name[0] == '80':
                    self.list_label.append(80)
                elif name[0] == '81':
                    self.list_label.append(81)
                elif name[0] == '82':
                    self.list_label.append(82)
                elif name[0] == '83':
                    self.list_label.append(83)
                elif name[0] == '84':
                    self.list_label.append(84)
                elif name[0] == '85':
                    self.list_label.append(85)
                elif name[0] == '86':
                    self.list_label.append(86)
                elif name[0] == '87':
                    self.list_label.append(87)
                elif name[0] == '88':
                    self.list_label.append(88)
                elif name[0] == '89':
                    self.list_label.append(89)
                else:
                    self.list_label.append(90)

        elif self.mode == 'test':
            dir = dir + '/test/'  # The test set path is "dir"/test/
            for file in os.listdir(dir):
                self.list_img.append(dir + file)  # Add the path to image list
                self.data_size += 1
                name = file.split(sep='.')

                if name[0] == '1':
                    self.list_label.append(1)  # Label is 0 when the picture is a age 1
                elif name[0] == '2':
                    self.list_label.append(2)
                elif name[0] == '3':
                    self.list_label.append(3)
                elif name[0] == '4':
                    self.list_label.append(4)
                elif name[0] == '5':
                    self.list_label.append(5)
                elif name[0] == '6':
                    self.list_label.append(6)
                elif name[0] == '7':
                    self.list_label.append(7)
                elif name[0] == '8':
                    self.list_label.append(8)
                elif name[0] == '9':
                    self.list_label.append(9)
                elif name[0] == '10':
                    self.list_label.append(10)
                elif name[0] == '11':
                    self.list_label.append(11)
                elif name[0] == '12':
                    self.list_label.append(12)
                elif name[0] == '13':
                    self.list_label.append(13)
                elif name[0] == '14':
                    self.list_label.append(14)
                elif name[0] == '15':
                    self.list_label.append(15)
                elif name[0] == '16':
                    self.list_label.append(16)
                elif name[0] == '17':
                    self.list_label.append(17)
                elif name[0] == '18':
                    self.list_label.append(18)
                elif name[0] == '19':
                    self.list_label.append(19)
                elif name[0] == '20':
                    self.list_label.append(20)
                elif name[0] == '21':
                    self.list_label.append(21)
                elif name[0] == '22':
                    self.list_label.append(22)
                elif name[0] == '23':
                    self.list_label.append(23)
                elif name[0] == '24':
                    self.list_label.append(24)
                elif name[0] == '25':
                    self.list_label.append(25)
                elif name[0] == '26':
                    self.list_label.append(26)
                elif name[0] == '27':
                    self.list_label.append(27)
                elif name[0] == '28':
                    self.list_label.append(28)
                elif name[0] == '29':
                    self.list_label.append(29)
                elif name[0] == '30':
                    self.list_label.append(30)
                elif name[0] == '31':
                    self.list_label.append(31)
                elif name[0] == '32':
                    self.list_label.append(32)
                elif name[0] == '33':
                    self.list_label.append(33)
                elif name[0] == '34':
                    self.list_label.append(34)
                elif name[0] == '35':
                    self.list_label.append(35)
                elif name[0] == '36':
                    self.list_label.append(36)
                elif name[0] == '37':
                    self.list_label.append(37)
                elif name[0] == '38':
                    self.list_label.append(38)
                elif name[0] == '39':
                    self.list_label.append(39)
                elif name[0] == '40':
                    self.list_label.append(40)
                elif name[0] == '41':
                    self.list_label.append(41)
                elif name[0] == '42':
                    self.list_label.append(42)
                elif name[0] == '43':
                    self.list_label.append(43)
                elif name[0] == '44':
                    self.list_label.append(44)
                elif name[0] == '45':
                    self.list_label.append(45)
                elif name[0] == '46':
                    self.list_label.append(46)
                elif name[0] == '47':
                    self.list_label.append(47)
                elif name[0] == '48':
                    self.list_label.append(48)
                elif name[0] == '49':
                    self.list_label.append(49)
                elif name[0] == '50':
                    self.list_label.append(50)
                elif name[0] == '51':
                    self.list_label.append(51)
                elif name[0] == '52':
                    self.list_label.append(52)
                elif name[0] == '53':
                    self.list_label.append(53)
                elif name[0] == '54':
                    self.list_label.append(54)
                elif name[0] == '55':
                    self.list_label.append(55)
                elif name[0] == '56':
                    self.list_label.append(56)
                elif name[0] == '57':
                    self.list_label.append(57)
                elif name[0] == '58':
                    self.list_label.append(58)
                elif name[0] == '59':
                    self.list_label.append(59)
                elif name[0] == '60':
                    self.list_label.append(60)
                elif name[0] == '61':
                    self.list_label.append(61)
                elif name[0] == '62':
                    self.list_label.append(62)
                elif name[0] == '63':
                    self.list_label.append(63)
                elif name[0] == '64':
                    self.list_label.append(64)
                elif name[0] == '65':
                    self.list_label.append(65)
                elif name[0] == '66':
                    self.list_label.append(66)
                elif name[0] == '67':
                    self.list_label.append(67)
                elif name[0] == '68':
                    self.list_label.append(68)
                elif name[0] == '69':
                    self.list_label.append(69)
                elif name[0] == '70':
                    self.list_label.append(70)
                elif name[0] == '71':
                    self.list_label.append(71)
                elif name[0] == '72':
                    self.list_label.append(72)
                elif name[0] == '73':
                    self.list_label.append(73)
                elif name[0] == '74':
                    self.list_label.append(74)
                elif name[0] == '75':
                    self.list_label.append(75)
                elif name[0] == '76':
                    self.list_label.append(76)
                elif name[0] == '77':
                    self.list_label.append(77)
                elif name[0] == '78':
                    self.list_label.append(78)
                elif name[0] == '79':
                    self.list_label.append(79)
                elif name[0] == '80':
                    self.list_label.append(80)
                elif name[0] == '81':
                    self.list_label.append(81)
                elif name[0] == '82':
                    self.list_label.append(82)
                elif name[0] == '83':
                    self.list_label.append(83)
                elif name[0] == '84':
                    self.list_label.append(84)
                elif name[0] == '85':
                    self.list_label.append(85)
                elif name[0] == '86':
                    self.list_label.append(86)
                elif name[0] == '87':
                    self.list_label.append(87)
                elif name[0] == '88':
                    self.list_label.append(88)
                elif name[0] == '89':
                    self.list_label.append(89)
                else:
                    self.list_label.append(90)
        else:
            print('Undefined Dataset!')

    # Reload the data.Dataset parent class method to get the content in the dataset
    def __getitem__(self, item):
        # Training set mode: Read The image and label
        if self.mode == 'train':
            # Open image
            img = Image.open(self.list_img[item])
            # Get the label corresponding to the image
            label = self.list_label[item]
            # Convert image and label into Tensor and return
            return self.transform(img), torch.LongTensor([label])

        # Test set mode: Read The image and label
        elif self.mode == 'validate':
            img = Image.open(self.list_img[item])
            # Get the label corresponding to the image
            label = self.list_label[item]
            # Convert image and label into Tensor and return
            return self.transform(img), torch.LongTensor([label])

        # Test set mode: Read The image and label
        elif self.mode == 'test':
            img = Image.open(self.list_img[item])
            # Get the label corresponding to the image
            label = self.list_label[item]
            # Convert image and label into Tensor and return
            return self.transform(img), torch.LongTensor([label])
        else:
            print('None')

    def __len__(self):
        # Returns the size of the data set
        return self.data_size

