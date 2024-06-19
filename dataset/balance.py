import os
from shutil import copyfile
import cv2

def balance_data(input_dir, output_dir, target_size, class_distribution):
    subsets = ['train', 'val', 'test']

    for subset in subsets:
        input_subset_dir = os.path.join(input_dir, subset)
        output_subset_dir = os.path.join(output_dir, subset)

        if not os.path.exists(output_subset_dir):
            os.makedirs(output_subset_dir)

        if subset in ['val', 'test']:
            # Copiar arquivos diretamente sem aumentação de dados
            for cls in class_distribution.keys():
                input_cls_dir = os.path.join(input_subset_dir, cls)
                output_cls_dir = os.path.join(output_subset_dir, cls)

                if not os.path.exists(output_cls_dir):
                    os.makedirs(output_cls_dir)

                files = os.listdir(input_cls_dir)

                for file in files:
                    src = os.path.join(input_cls_dir, file)
                    dst = os.path.join(output_cls_dir, file)
                    copyfile(src, dst)
        else:
            classes = class_distribution.keys()
            max_samples = max([class_distribution[c] for c in classes])

            for cls in classes:
                input_cls_dir = os.path.join(input_subset_dir, cls)
                output_cls_dir = os.path.join(output_subset_dir, cls)

                if not os.path.exists(output_cls_dir):
                    os.makedirs(output_cls_dir)

                files = os.listdir(input_cls_dir)

                num_samples = len(files)
                if num_samples < max_samples:
                    print(f"Aumentando dados para a classe {cls}")
                    target_samples = max_samples - num_samples
                    augment_data(input_cls_dir, output_cls_dir, target_samples, target_size,32)

                for file in files:
                    src = os.path.join(input_cls_dir, file)
                    dst = os.path.join(output_cls_dir, file)
                    copyfile(src, dst)

def augment_data(input_dir, output_dir, target_samples, target_size, batch_size):
    augmenter = iaa.Sequential([
        iaa.Rotate((-40, 40)),
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.Resize(target_size)
    ])

    files = os.listdir(input_dir)

    iterations = target_samples // len(files) + 1

    print(f"Iterations: {iterations}")

    for i in range(iterations):
        batch = []
        for file in files:
            img_path = os.path.join(input_dir, file)
            img = cv2.imread(img_path)
            batch.append(img)

        batch_augmented = augmenter(images=batch)

        for img_augmented, file in zip(batch_augmented, files):
            img_augmented = cv2.cvtColor(img_augmented, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(output_dir, f"aug_{file}"), img_augmented)

    print(f"{target_samples} imagens aumentadas para a classe {os.path.basename(input_dir)}")

class_distribution = {
    '0': 1407, '1': 281, '2': 767, '3': 142, '4': 222
}

input_dir = '/content/drive/MyDrive/Faculdade/TCC/src/APTOS_processed_256'
output_dir = '/content/drive/MyDrive/Faculdade/TCC/src/APTOS_processed_256_balanced_augmented'
target_size = (256, 256)

balance_data(input_dir, output_dir, target_size, class_distribution)