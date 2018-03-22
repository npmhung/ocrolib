from io import open

jis1map = None
jis2map = None

def init_char_map(model_dir):
    global jis1map
    jis1map = open(model_dir + '/data/katakana_map', encoding='utf-8').readlines()

def init_char_map_kanji(model_dir):
    global jis2map
    file_path = model_dir + '/data/etl9_map'
    f = open(file_path, 'r')
    table = {}
    for i, line in enumerate(f):
        try:
            fields = line.strip().split(' ')
            table[int(fields[0])] = fields[-1].strip()[-4:]
        except ValueError:
            print("Read table error at line ", i)
            pass

    f.close()
    jis2map = table

def label2unicode_etl9(label):
    jis_code = jis2map[label]
    b = b'\033$B' + bytearray.fromhex(jis_code)
    c = b.decode('iso2022_jp')
    return c

def jis2unicode(jis_code):
    # print(jis_code)
    try:
        jis_code2=hex(jis_code)[2:]
        b = b'\033$B' + bytes.fromhex(jis_code2)
        c = b.decode('iso2022_jp')
        return c
    except Exception:
        for i,l in enumerate(jis1map):
            if i%2==1:
                # print(l.rstrip())
                if int(jis_code)==int(l.rstrip()):
                    return jis1map[i-1].strip()
    return ''