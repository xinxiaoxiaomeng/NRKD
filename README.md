
This code implements a new similarity-based knowledge distillation method that transfers neighbourhood relationship knowledge by selecting K-nearest neighbours for each sample.

## Run

 ```
 #training a teacher model:
 python train_teacher.py --model resnet32x4 --dataset cifar100 --epochs 240 --lr_decay_epochs 150,180,210 --learning_rate 0.05 --batch_size 64
 
 ```


 ```
 #training a student model:
 python train_student.py --teacher resnet32x4 --checkpoints checkpoints/cifar100/resnet32x4.pth --student resnet8x4 --dataset cifar100 --epochs 240 --lr_decay_epochs 150,180,210 --learning_rate 0.05 --batch_size 64

```
