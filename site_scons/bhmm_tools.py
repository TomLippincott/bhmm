from SCons.Builder import Builder
from SCons.Script import *
import re
from glob import glob
from functools import partial
import logging
import os.path
import os
import cPickle as pickle
import numpy
import math
try:
    import lxml.etree as et
except:
    import xml.etree.ElementTree as et
import xml.sax
import sys
import gzip
from os.path import join as pjoin
from os import listdir
import tarfile
from random import randint
from common_tools import DataSet, meta_open

def bad_word(w):
    return (w.startswith("*") and w.endswith("*")) or (w.startswith("<") and w.endswith(">")) or (w.startswith("(") and w.endswith(")"))

def load_analyses(fname):
    retval = {}
    with meta_open(fname) as ifd:
        for i, l in enumerate(ifd):
            word, analyses = l.split("\t")
            retval[word] = (i, set())
            for a in analyses.split(","):
                i, ss = retval[word]
                ss.add(tuple([x.split(":")[0].strip() for x in a.split() if not x.startswith("~")]))
                retval[word] = (i, ss)
    return retval

def generic_to_bhmm(target, source, env, get_sentences=None, get_analyses=lambda x : {}):
    preamble_tb = et.TreeBuilder()
    datasets_tb = et.TreeBuilder()

    analyses = get_analyses(source)
    tagToId = {}
    wordToId = {}
    analysisToId = {}
    datasets_tb.start("sentences", {})
    if get_sentences:
        for i, s in enumerate([x for x in get_sentences(source) if len(x) > 0]):
            datasets_tb.start("sentence", {"n" : str(i + 1)})
            for j, (word, tag) in enumerate(s):
                wordToId[word] = wordToId.get(word, len(wordToId))
                datasets_tb.start("location", {"n" : str(j + 1)})
                datasets_tb.start("word", {"id" : str(wordToId[word])}), datasets_tb.end("word")
                if word in analyses:
                    datasets_tb.start("analyses", {})
                    for a in analyses[word][1]:
                        analysisToId[a] = analysisToId.get(a, len(analysisToId))
                        datasets_tb.start("analysis", {"id" : str(analysisToId[a])}), datasets_tb.end("analysis")
                    datasets_tb.end("analyses")
                if tag:
                    tagToId[tag] = tagToId.get(tag, len(tagToId))
                    datasets_tb.start("tag", {"id" : str(tagToId[tag])}), datasets_tb.end("tag")
                datasets_tb.end("location")
            datasets_tb.end("sentence")
    datasets_tb.end("sentences")
    
    preamble_tb.start("preamble", {})
    preamble_tb.start("analysis_inventory", {})
    for analysis, i in sorted(analysisToId.iteritems(), lambda x, y : cmp(x[1], y[1])):
        preamble_tb.start("entry", {"id" : str(i)})
        for m in analysis:
            preamble_tb.start("morph", {})
            try:
                preamble_tb.data(m.decode("utf-8"))
            except:
                preamble_tb.data(m)
            preamble_tb.end("morph")
        preamble_tb.end("entry")
    preamble_tb.end("analysis_inventory")
    preamble_tb.start("tag_inventory", {})
    for tag, i in sorted(tagToId.iteritems(), lambda x, y : cmp(x[1], y[1])):
        preamble_tb.start("entry", {"id" : str(i)})
        try:
            preamble_tb.data(tag.decode("utf-8"))
        except:
            preamble_tb.data(tag)
        preamble_tb.end("entry")
    preamble_tb.end("tag_inventory")
    preamble_tb.start("word_inventory", {})
    for word, i in sorted(wordToId.iteritems(), lambda x, y : cmp(x[1], y[1])):
        preamble_tb.start("entry", {"id" : str(i)})
        try:
            preamble_tb.data(word.decode("utf-8"))
        except:
            preamble_tb.data(word)
        preamble_tb.end("entry")
    preamble_tb.end("word_inventory")
    preamble_tb.end("preamble")
    tb = et.TreeBuilder()
    tb.start("xml", {})
    tb.end("xml")
    xml = tb.close()
    xml.append(preamble_tb.close())
    xml.append(datasets_tb.close())
    with meta_open(target[0].rstr(), "w") as ofd:
        ofd.write(et.tounicode(tb.close(), pretty_print=True).encode("utf-8"))
    return None

def dummy(sources):
    return []

#
# Stanardized data sets
#
def standardized_sentences(sources):
    with meta_open(sources[0].rstr()) as ifd:
        return [[(w, None) for w in l.strip().split()] for l in ifd]

#
# BABEL data sets
#
# def babel_limited_training(sources):
#     retval = []
#     with tarfile.open(sources[0].rstr(), "r:gz") as ifd:
#         for fname in [x for x in ifd.getnames() if re.match(r".*sub-train/transcription.*.txt*", x)]:
#             retval += [[(w, None) for w in l.split() if not bad_word(w)] for l in ifd.extractfile(fname) if not l.startswith("[")]
#     return retval

# def babel_full_training(sources):
#     retval = []
#     #with tarfile.open(sources[0].rstr(), "r:gz") as ifd:
#     #    for fname in [x for x in ifd.getnames() if re.match(r".*training/transcription.*.txt*", x)]:
#     #        retval += [[(w, None) for w in l.split() if not bad_word(w)] for l in ifd.extractfile(fname) if not l.startswith("[")]
#     return retval

#def babel_development(sources):
#    retval = []
#    with tarfile.open(sources[0].rstr(), "r:gz") as ifd:
#        for fname in [x for x in ifd.getnames() if re.match(r".*dev/transcription.*.txt*", x)]:
#            retval += [[(w, None) for w in l.split() if not bad_word(w)] for l in ifd.extractfile(fname) if not l.startswith("[")]
#    return retval

def generic_analyses(sources):
    retval = {}
    if len(sources) == 1:
        return {}
    else:
        with meta_open(sources[-1].rstr()) as ifd:
            for l in ifd:
                try:
                    word, rest = l.strip().split("\t")
                except:
                    continue
                aa = set()
                for a in rest.split(","):
                    ms = [y for y in [m.split(":")[0] for m in a.split()] if y != "~"]
                    if "".join(ms) == word:
                        aa.add(tuple(ms))
                retval[word] = (len(retval), aa)
    return retval

#
# Ukwabelana data set
#
def ukwabelana_analyses(sources):
    analyses = {}
    with tarfile.open(sources[0].rstr()) as ifd:
        for i, l in enumerate(ifd.extractfile("2010.07.17.WordListSegmented.txt")):
            toks = l.split(",")
            first = toks[0].split()
            w = first[0]
            ps = [first[1:]] + [x.split() for x in toks[1:]]
            analyses[w] = (i, set([tuple(x) for x in ps]))
    return analyses

def ukwabelana_sentences(sources):
    with tarfile.open(sources[0].rstr()) as ifd:
        return [[x.split("_") for x in l.strip().split()] for l in ifd.extractfile("2010.07.17.ZuluSentencesPOStagged.txt")]

#
# PENN Treebank data set
#
def penn_sentences(sources):
    with meta_open(sources[0].rstr()) as ifd:
        return [[["/".join(y[:-1]), y[-1]] for y in [x.split("/") for x in l.strip().split()]] for l in ifd]

#
# UPC data set
#
def upc_sentences(sources):
    with tarfile.open(sources[0].rstr()) as ifd:      
        return [[y for y in [re.sub(r"\t+", "\t", l.strip()).split("\t") for l in s.strip().split("\n")] if len(y) == 2] for s in ifd.extractfile("UPEC.txt").read().split("\n\n")]

#
# TIGER data set
#
def tiger_sentences(sources):
    retval = []
    with tarfile.open(sources[0].rstr()) as ifd:
        for s in et.parse(ifd.extractfile("tiger_release_aug07.corrected.16012013.xml")).getiterator("terminals"):
            retval.append([(l.get("word"), l.get("pos")) for l in s.getiterator("t")])
    return retval

# def dataset_to_emma(target, source, env):
#     with meta_open(source[0].rstr()) as ifd:
#         data = DataSet.from_stream(ifd)
#     with meta_open(target[0].rstr(), "w") as ofd:
#         for word, analyses in sorted(data.get_analyses().iteritems()):
#             x = "%s\t%s\n" % (word, ", ".join([" ".join(["%s:NULL" % m[1] for m in a]) for a in analyses]))
#             ofd.write(x.encode("utf-8"))
#     return None


def morfessor_to_tripartite(target, source, env):
    with meta_open(source[0].rstr()) as ifd:
        data = DataSet.from_stream(ifd)
    for k in data.indexToAnalysis.keys():
        old = [x[1] for x in data.indexToAnalysis[k]]
        sizes = sorted(enumerate([len(x) for x in old]), lambda x, y : cmp(x[1], y[1]))
        stem_index = sizes[-1][0]
        stem = ("stem", old[stem_index])
        prefix = ("prefix", "".join(old[:stem_index]))
        suffix = ("suffix", "".join(old[stem_index + 1:]))        
        data.indexToAnalysis[k] = [({}, x[1]) for x in [prefix, stem, suffix] if x[1] != ""]
    with meta_open(target[0].rstr(), "w") as ofd:
        data.write(ofd)
    return None

def xml_to_sadegh(target, source, env):
    analyses = {}
    counts = {}
    lookup = {"PRE" : "a", "STM" : "b", "SUF" : "c"}
    with meta_open(source[0].rstr()) as ifd:
        #data = DataSet.from_stream(ifd)[0]
        #for i, a in data.indexToAnalysis.iteritems():
        #    morphs = [(x.text, lookup[x.get("type")[0:3]]) for m in a]
        xml = et.parse(ifd)
        for i in xml.findall("//analysis_inventory/entry"):
            index = int(i.get("id"))
            morphs = [(x.text, lookup[x.get("type")[0:3]]) for x in i.findall("morph")]
            if any([re.match(r".*\d.*", x) for (x, a) in morphs]):
                continue
            intervals = [len(y) for y in re.match(r"^(a*)([^c]+)(.*?)$", "".join([x[1] for x in morphs])).groups()]            
            prefix = "".join([x[0] for x in morphs[0:intervals[0]]])
            stem = "".join([x[0] for x in morphs[intervals[0]:intervals[0] + intervals[1]]])
            suffix = "".join([x[0] for x in morphs[intervals[0] + intervals[1]:]])
            if prefix == "":
                prefix = "<epsilon>"
            if suffix == "":
                suffix = "<epsilon>"
            analyses[index] = (prefix, stem, suffix)
        for l in xml.findall("//location"):
            t = int(l.find("tag").get("id"))
            a = int(l.find("analyses/analysis").get("id"))
            if a not in analyses:
                continue
            counts[(t, a)] = counts.get((t, a), 0) + 1
            pass
    #     if env.get("LIMITED", False):
    #         data = xml.findall("//dataset[@type='limited']//analysis")
    #     else:
    #         data = xml.findall("//dataset//analysis")
    #     for a in data:
    #         index = int(a.get("id"))
    #         counts[index] = counts.get(index, 0) + 1
    with meta_open(target[0].rstr(), "w") as ofd:
        ofd.write("\n".join(["%d\t%d\t%s" % (c, t + 1, "\t".join(analyses[a]).encode("utf-8")) for (t, a), c in counts.iteritems()]))
    return None

def xml_to_sadegh_emitter(target, source, env):
    #path, name = os.path.split(source[0].rstr())
    #new_name = "%s_for_sadegh.txt" % name.split(".")[0]
    return target, source #os.path.join(path, new_name), source

def collate_results(target, source, env):
    data = {}
    for k, v in env["MORPHOLOGY_RESULTS"].iteritems():
        with meta_open(v[0].rstr()) as ifd:
            names, values = [x.strip().split("\t") for x in ifd][0:2]
            data[k] = {n : "%.3f" % float(v) for n, v in zip(names, values)}
    for k, v in env["TAGGING_RESULTS"].iteritems():
        with meta_open(v[0].rstr()) as ifd:
            names, values = [x.strip().split("\t") for x in ifd][0:2]
            if k in data:
                data[k].update({n : "%.3f" % float(v) for n, v in zip(names, values)})
            else:
                data[k] = {n : "%.3f" % float(v) for n, v in zip(names, values)}
    with meta_open(target[0].rstr(), "w") as ofd:
        properties = ["Lang", "Method", "Units"]
        names = sorted(set(sum([x.keys() for x in data.values()], [])))
        ofd.write("\t".join(properties + names) + "\n")
        for k, v in data.iteritems():
            k = {"METHOD" : k[1], "LANG" : k[0], "UNITS" : k[2].split("-")[0]}
            ofd.write("\t".join([k.get(p.upper(), "").title() for p in properties] + [v.get(n, "") for n in names]) + "\n")
    return None

def create_subset(target, source, env):
    amount = source[1].read()
    method = source[2].read()
    with meta_open(source[0].rstr()) as ifd:
        data = DataSet.from_stream(ifd)
    subset = data.get_subset(range(amount))
    with meta_open(target[0].rstr(), "w") as ofd:
        subset.write(ofd)
    return None

def top_words_by_tag(target, source, env):
    with meta_open(source[0].rstr()) as ifd:
        data = DataSet.from_stream(ifd)[-1]
    counts = numpy.zeros(shape=(len(data.indexToWord), len(data.indexToTag)))
    for sentence in data.sentences:
        for w, t, aa in sentence:
            counts[w, t] += 1
    tag_totals = counts.sum(0)
    word_totals = counts.sum(1)
    keep = 10
    with meta_open(target[0].rstr(), "w") as ofd:
        for tag_id, tag_total in enumerate(tag_totals):
            word_counts = counts[:, tag_id] #.argsort()
            indices = [(i, word_counts[i]) for i in reversed(word_counts.argsort())][0:keep]
            ofd.write(" ".join(["%s-%.2f-%.2f" % (data.indexToWord[i], float(c) / tag_total, float(c) / word_totals[i]) for i, c in indices]) + "\n")
    return None

def conllish_to_xml(target, source, env):
    with meta_open(source[0].rstr()) as ifd:
        sentences = [[(w, t, []) for w, t in [re.split(r"\s+", x) for x in s.split("\n") if not re.match(r"^\s*$", x)]] for s in re.split(r"\n\n", ifd.read(), flags=re.M)]
    data = DataSet.from_sentences(sentences)
    with meta_open(target[0].rstr(), "w") as ofd:
        data.write(ofd)
    return None

def TOOLS_ADD(env):
    env.Append(BUILDERS = {
            "CONLLishToXML" : Builder(action=conllish_to_xml),
            "RunScala" : Builder(action="${SCALA_BINARY} ${SCALA_OPTS} -cp ${SOURCES[0]} ${SOURCES[1]} --output ${TARGETS[0]} --input ${SOURCES[2]} ${ARGUMENTS}"),
            "UkwabelanaToBHMM" : Builder(action=partial(generic_to_bhmm, get_sentences=ukwabelana_sentences, get_analyses=ukwabelana_analyses)),
            "PennToBHMM" : Builder(action=partial(generic_to_bhmm, get_sentences=penn_sentences, get_analyses=generic_analyses)),
            "UpcToBHMM" : Builder(action=partial(generic_to_bhmm, get_sentences=upc_sentences)),
            "TigerToBHMM" : Builder(action=partial(generic_to_bhmm, get_sentences=tiger_sentences)),
            "StandardizedToBHMM" : Builder(action=partial(generic_to_bhmm, get_sentences=standardized_sentences, get_analyses=generic_analyses)),
            #"BabelToBHMM" : Builder(action=partial(generic_to_bhmm, 
            #                                       get_limited_training=babel_limited_training, 
            #                                       get_full_training=babel_full_training,
            #                                       get_development=babel_development,
            #                                       get_analyses=generic_analyses)),                                                   
            #"EvaluateMorphology" : Builder(action="python bin/EMMA2.py -g ${SOURCES[0]} -p ${SOURCES[1]} > ${TARGET}"),
            #"DatasetToEmma" : Builder(action=dataset_to_emma),

            "XMLToSadegh" : Builder(action=xml_to_sadegh, emitter=xml_to_sadegh_emitter),
            "CollateResults" : Builder(action=collate_results),
            "CreateSubset" : Builder(action=create_subset),
            "MorfessorToTripartite" : Builder(action=morfessor_to_tripartite),
            "TopWordsByTag" : Builder(action=top_words_by_tag),
            })
