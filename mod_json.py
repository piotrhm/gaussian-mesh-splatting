import json
import io

json_file = '/home/gmhelm/repo/gaussian-mesh-splatting/data/person_1/transforms_train.json'

try:
    to_unicode = unicode
except NameError:
    to_unicode = str

frame_new = []
with open(json_file) as f:
    d = json.load(f)
    for meta_data in d['frames']:
        if "0" in meta_data['file_path'][-4:-3]:
            frame_new.append(meta_data)

    d['frames'] = frame_new
    with io.open('data.json', 'w', encoding='utf8') as outfile:
        str_ = json.dumps(d,
                        indent=4, sort_keys=True,
                        separators=(',', ': '), ensure_ascii=False)
        outfile.write(to_unicode(str_))

