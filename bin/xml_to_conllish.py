import lxml.etree as et
from common_tools import DataSet, meta_open
import argparse
import os.path

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", dest="input")
parser.add_argument("-o", "--output", dest="output")
options = parser.parse_args()

sizes = {"train" : 50000, 
         "dev" : 8000, 
         "test" :8000,
         #"small_eval" : 100000,
         }

indices = {}

with meta_open(options.input) as ifd:
    data = DataSet.from_stream(ifd)
    start = 0
    end = 0
    for name, size in sizes.iteritems():        
        total_words = 0
        while total_words < size:
            total_words += len(data.sentences[end])
            end += 1
        indices[name] = (start, end)
        start = end
    indices["big_eval"] = (end, len(data.sentences))
        
for name in ["train", "dev", "test"]:
    start, end = indices[name]
    with meta_open(os.path.join(options.output, "pos", "%s.pos" % name), "w") as ofd:
        text = "\n\n".join(["\n".join(["%s\t%s" % (data.indexToWord[w].lower(), data.indexToTag[t]) for w, t, aa in s]) for s in data.sentences[start:end]])
        ofd.write(text.encode("utf-8"))
    with meta_open(os.path.join(options.output, "pos_cap", "%s.pos" % name), "w") as ofd:
        text = "\n\n".join(["\n".join(["%s\t%s" % (data.indexToWord[w], data.indexToTag[t]) for w, t, aa in s]) for s in data.sentences[start:end]])
        ofd.write(text.encode("utf-8"))

# for name in ["small_eval", "big_eval"]:
#     continue
#     start, end = indices[name]
#     counts = {}
#     for s in data.sentences[start:end]:
#         for w, t, aa in s:
#             word = data.indexToWord[w].lower()
#             counts[word.lower()] = counts.get(word.lower(), 0) + 1
#     with meta_open(os.path.join(options.output, "oov_eval", "%s.freq" % name), "w") as ofd:
#         text = "\n".join(["%d %s" % (v, k) for k, v in counts.iteritems()])
#         ofd.write(text.encode("utf-8"))
