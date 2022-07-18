from old_dataset_processing import Old_Processing
import torchvision.transforms as transforms
import argparse
import os

if __name__ == "__main__":
    # python old.py imagenet_10/train/ 4 imagenet_10/old_compiled_new imagenet_10/ckpt.pth 50 10
    # parser = argparse.ArgumentParser()
    # parser.add_argument("ds_path", help="path to old dataset")
    # parser.add_argument("gpu", help="gpu to use")
    # parser.add_argument("out_dir", help="where output should be saved")
    # parser.add_argument("weights", help="path to weights file, if using pretrained resnet50 you can type anything")
    # parser.add_argument("architecture", help="version of resnet, options are 18, 34, 50, 101, 152 or pretrained for a pretrained resnet50")
    # parser.add_argument("classes", help="num of classes in the final layer of the model")
    # args = parser.parse_args()
    old_transforms = transforms.Compose([
        transforms.Resize(size=(150,150)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])

    # op = Old_Processing(args.ds_path, args.gpu, args.out_dir, args.weights, args.architecture, args.classes, old_transforms)
    ds_path = os.path.join("./dataset")
    gpu = "0"
    weights = os.path.join("model_lib", "src", "unbalanced_model.pth")
    arch = "50"
    classes = "3"
    # weights = os.path.join("resnet101_weights.pth")
    # arch = "101"
    # classes = "1000"    
    # out = "/output"
    out = "./output-new"
    if not os.path.exists(os.path.join(out)):
        os.makedirs(out)
    op = Old_Processing(ds_path, gpu, out, weights, arch, classes, old_transforms)
    op.extract()
    op.distance()
    # op.compress()

