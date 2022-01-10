# Real-time-Age-Prediction

## Dataset

- The age is annotated in the file name as "age.#.jpg" </br>
<img src="https://github.com/Ericdiii/Real-time-Age-Prediction/blob/train/image/data1.png?raw=true" height="130"/> <img src="https://github.com/Ericdiii/Real-time-Age-Prediction/blob/train/image/data4.png?raw=true" height="130"/>  </br>
<img src="https://github.com/Ericdiii/Real-time-Age-Prediction/blob/train/image/data2.png?raw=true" height="123"/> <img src="https://github.com/Ericdiii/Real-time-Age-Prediction/blob/train/image/data3.png?raw=true" height="130"/>  </br>

- Create new folder to save your own data
* data
  * train
  * test
  * validate 

- Public dataset: [UTKFace](https://susanqq.github.io/UTKFace/)

## Train

- Create new folder to save your model
* model
  * SSRNet </br>
<img src="https://github.com/Ericdiii/Real-time-Age-Prediction/blob/train/image/folder.png?raw=true" height="150"/> 

- Use your dataset to train the model
```sh
python SSRNet_train.py
```

## Test

- Test the trained model with your test set
```sh
python SSRNet_test.py
```
