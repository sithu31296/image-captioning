import json
import argparse
from tqdm import tqdm


def main(input_txt, output_json):
    annot_format = {
        'info': {
            'year': 2014, 
            'version': '1.0',
            'description': 'This is stable 1.0 version of the 2014 MS COCO dataset.', 
            'contributor': 'Microsoft COCO group',
            'url': 'http://mscoco.org', 
            'date_created': '2015-01-27 09:11:52.357475'
        }, 
        'licenses': [
            {
                'url': 'http://creativecommons.org/licenses/by-nc-sa/2.0/', 
                'id': 1, 
                'name': 'Attribution-NonCommercial-ShareAlike License'
            }, 
            {
                'url': 'http://creativecommons.org/licenses/by-nc/2.0/', 
                'id': 2, 
                'name': 'Attribution-NonCommercial License'
            }, 
            {
                'url': 'http://creativecommons.org/licenses/by-nc-nd/2.0/', 
                'id': 3, 
                'name': 'Attribution-NonCommercial-NoDerivs License'
            }, 
            {
                'url': 'http://creativecommons.org/licenses/by/2.0/', 
                'id': 4, 
                'name': 'Attribution License'
            }, 
            {
                'url': 'http://creativecommons.org/licenses/by-sa/2.0/', 
                'id': 5, 
                'name': 'Attribution-ShareAlike License'
            }, 
            {   
                'url': 'http://creativecommons.org/licenses/by-nd/2.0/', 
                'id': 6, 
                'name': 'Attribution-NoDerivs License'
            }, 
            {
                'url': 'http://flickr.com/commons/usage/', 
                'id': 7, 
                'name': 'No known copyright restrictions'
            }, 
            {
                'url': 'http://www.usa.gov/copyright.shtml', 
                'id': 8, 
                'name': 'United States Government Work'
            }
        ], 
        'type': 'captions',
        'images': [],
        'annotations': []
    }

    with open(input_txt) as f:
        lines = f.read().splitlines()

    for i, line in tqdm(enumerate(lines), total=len(lines)):
        fname, caption = line.split(' ', maxsplit=1)
        annot_format['images'].append({
            "id": i,
            "width": 512,
            "height": 512,
            "filename": fname,
            "license": 1,
            "flickr_url": '',
            "coco_url": '',
            "date_captured": ''
        })

        annot_format['annotations'].append({
            "id": i,
            "image_id": i,
            "caption": caption
        })

    with open(output_json, 'w') as f:
        json.dump(annot_format, f)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-txt', type=str, default='data/val.txt')
    parser.add_argument('--output-json', type=str, default='data/conceptual_coco_format.json')
    args = parser.parse_args()

    main(args.input_txt, args.output_json)