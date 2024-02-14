import torch
from torchvision import transforms
import argparse
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import os
from tqdm import tqdm

from cv_models import DEVICE, LOCAL, vgg
import dataset


def train(running_on, model, model_name, dataset_name, train_dataset, train_loader, val_dataset, val_loader):
    model = model.to(DEVICE)
    EPOCHS = 100
    BEST_ACCURACY = -10.0
    WEIGHT_SAVE_PATH = running_on['weights_save_path']

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    print('Total training samples:', len(train_dataset))
    print('Total index samples:', len(train_loader))

    for epoch in range(EPOCHS):
        print('-' * 30 + 'begin EPOCH ' + str(epoch + 1) + '-' * 30)
        model.train()
        running_loss = 0.0

        for batch, data in enumerate(tqdm(train_loader)):
            images, labels, _ = data
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            out = model(images)

            loss = loss_fn(out, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # if (batch + 1) % 100 == 0:
            #     print('Training Epoch: %d, batch_idx:%5d, loss: %.8f' % (
            #     epoch + 1, batch + 1, running_loss / images.shape[0]))
            #     running_loss = 0.0


            break
        # validation after finish One EPOCH training
        model.eval()
        val_loss = 0.0
        num_correct = 0
        with torch.no_grad():
            for data in tqdm(val_loader):
                images, labels, _ = data
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                out = model(images)
                loss = loss_fn(out, labels)
                _, pred = torch.max(out, 1)
                num_correct += (pred == labels).sum()
                val_loss += loss.item()
                break

            val_accuracy = num_correct / len(val_dataset)
            print('Val Loss:{:.6f}, accuracy:{:.10f}'.format(val_loss, val_accuracy))
            print('*' * 50)

        # 如果模型表现效果好，则保存
        if val_accuracy > BEST_ACCURACY:
            model_save_dir = WEIGHT_SAVE_PATH

            # 删除已经有的文件,只保留n+1个模型
            num_saved = 2
            all_weights_temp = os.listdir(model_save_dir)
            all_weights = []
            for weights in all_weights_temp:
                if weights.endswith('.pth'):
                    all_weights.append(weights)
            if len(all_weights) > num_saved:
                sorted = []
                for weight in all_weights:
                    acc = weight.split('-')[-1]
                    sorted.append((weight, acc))
                sorted.sort(key=lambda w: w[1], reverse=True)
                del_path = os.path.join(model_save_dir, sorted[-1][0])
                os.remove(del_path)
                print('del file:', del_path)

            # 保存新的模型
            save_name = f"{model_name}-{dataset_name}-{epoch + 1:03d}-{val_accuracy:.8f}.pth"
            save_path = os.path.join(model_save_dir, save_name)

            # 存储权重
            torch.save(model.state_dict(), save_path)

            BEST_ACCURACY = val_accuracy
            print(f'{model_name} saved on epoch:{epoch + 1}, current BEST_ACCURACY:{BEST_ACCURACY}')
            print('*' * 50)


if __name__ == '__main__':
    BATCH_SIZE = 4
    running_on = LOCAL
    model = vgg.vgg16()
    # weights_path = r'D:\chrom_download\vggonECPD-050-0.97701919.pth'
    # model.load_state_dict(torch.load(weights_path, map_location=torch.device(DEVICE)))

    train_dataset = dataset.MyDataset(running_on, dataset_name='D3', txt_name='train.txt', transformer_mode=0)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_dataset = dataset.MyDataset(running_on, dataset_name='D3', txt_name='val.txt', transformer_mode=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    train(running_on=running_on,
          model=model, model_name='vgg',
          dataset_name='ECPD',
          train_dataset=train_dataset,
          train_loader=train_loader,
          val_dataset=val_dataset,
          val_loader=val_loader
          )

    # test_dataset = dataset.MyDataset(running_on=running_on, txt_name='test.txt', transformer_mode=0)
    # test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    #
    # model.eval()

    # # 预测结果和真实标签
    # y_pred = []
    # y_true = []
    # num_correct = 0
    #
    # with torch.no_grad():
    #     for data in tqdm(test_loader):
    #         images, labels, _ = data
    #         images = images.to(DEVICE)
    #         labels = labels.to(DEVICE)
    #         out = model(images)
    #         _, pred = torch.max(out, 1)
    #         num_correct += (pred == labels).sum()
    #         # 将label和pred加入列表中
    #         y_pred.extend(pred.cpu().numpy())
    #         y_true.extend(labels.cpu().numpy())
    #
    #     val_accuracy = num_correct / len(test_dataset)
    #
    #     print('Acc on test set:', val_accuracy)
    #
    #     cm = confusion_matrix(y_true, y_pred)
    #     print("cm:\n", cm)















