import json


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
            json.dump(self.annotation_obj, f, indent=4)

        with open(result_file, 'w') as f:
            print('dumping {} results to {}'.format(len(self.result_obj), result_file))
            json.dump(self.result_obj, f, indent=4)

