import argparse
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def evaluate(truth_json_name, submit_json_name):
    cocoGt = COCO(truth_json_name)
    cocoDt = cocoGt.loadRes(submit_json_name)

    cocoEval = COCOeval(cocoGt, cocoDt, 'segm')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", type=str, default="dataset/train.json",
                        help="path to the ground truth json file")
    parser.add_argument("--submit", type=str, default="saved/mRCNN_0_3/results/maskRCNN_134_train_0.json",
                        help="path to the submission json file")
    opt = parser.parse_args()
    print(opt)

    evaluate(opt.truth, opt.submit)
