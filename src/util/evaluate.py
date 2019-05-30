import json
import traceback

try:
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap
except:
    traceback.print_exc()
    print('import from coco-caption failed')

try:
    from .customjson import dump_custom
    dump_func = dump_custom
except:
    dump_func = json.dump


class COCOResultGenerator:
    def __init__(self):
        self.result_obj = []
        self.annotation_obj = {'info': 'N/A', 'licenses': 'N/A', 'type': 'captions', 'images': [], 'annotations': []}
        self.caption_id = 0
        self.annotation_image_set = set()
        self.test_image_set = set()

    def add_annotation(self, image_id, caption_raw):
        if image_id not in self.annotation_image_set:
            self.annotation_obj['images'].append({'id': image_id})
            self.annotation_image_set.add(image_id)
        self.annotation_obj['annotations'].append({'image_id': image_id, 'caption': caption_raw, 'id': self.caption_id})
        self.caption_id += 1

    def add_output(self, image_id, caption_output, image_filename=None, metadata=None):
        assert(image_id in self.annotation_image_set and image_id not in self.test_image_set)
        item = {"image_id": image_id, "caption": caption_output}
        if metadata is not None:
            item['meta'] = metadata
        if image_filename is not None:
            item["image_filename"] = image_filename
        self.result_obj.append(item)
        self.test_image_set.add(image_id)

    def has_output(self, image_id):
        return image_id in self.test_image_set

    def get_annotation_and_output(self):
        return self.annotation_obj, self.result_obj

    def dump_annotation_and_output(self, annotation_file, result_file):
        with open(annotation_file, 'w') as f:
            print('dumping {} annotations to {}'.format(len(self.annotation_obj['annotations']), annotation_file))
            dump_func(self.annotation_obj, f, indent=4)

        with open(result_file, 'w') as f:
            print('dumping {} results to {}'.format(len(self.result_obj), result_file))
            dump_func(self.result_obj, f, indent=4)


def eval(ann_file, res_file):
    coco = COCO(ann_file)
    cocoRes = coco.loadRes(res_file)
    # create cocoEval object by taking coco and cocoRes
    # cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval = COCOEvalCap(coco, cocoRes, use_scorers=['Bleu', 'METEOR', 'ROUGE_L', 'CIDEr'])

    # evaluate on a subset of images by setting
    # cocoEval.params['image_id'] = cocoRes.getImgIds()
    # please remove this line when evaluating the full validation set
    cocoEval.params['image_id'] = cocoRes.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    cocoEval.evaluate()

    all_score = {}
    # print output evaluation scores
    for metric, score in cocoEval.eval.items():
        # print('%s: %.4f' % (metric, score))
        all_score[metric] = score

    img_scores = [cocoEval.imgToEval[key] for key in cocoEval.imgToEval.keys()]

    return all_score