import os
import os.path
import sys
from SCons.Tool import textfile
from glob import glob
import logging
import time
import re
import bhmm_tools
from os.path import join as pjoin

vars = Variables("custom.py")
vars.AddVariables(
    ("OUTPUT_WIDTH", "", 130),
    BoolVariable("DEBUG", "", False),
    BoolVariable("TEST", "", False),
    ("MORPHOLOGY_PATH", "", False),
    ("LANGUAGES", "", {}),
    ("LANGUAGE_PACKS", "", {}),

    EnumVariable("MODE", "", "morphology", ["joint", "morphology", "tagger"]),

    ("NUM_SENTENCES", "", -1),
    ("NUM_TAGS", "", 17),
    ("NUM_BURNINS", "", 10),
    ("NUM_SAMPLES", "", 1),
    ("MARKOV", "", 2),

    ("TRANSITION_PRIOR", "", .1),
    ("EMISSION_PRIOR", "", .1),

    BoolVariable("SYMMETRIC_TRANSITION_PRIOR", "", False),
    BoolVariable("SYMMETRIC_EMISSION_PRIOR", "", False),

    ("OPTIMIZE_EVERY", "", 0),
    ("ANNEALING", "", 1.0),

    ("VARIATION_OF_INFORMATION", "", 0),
    ("PERPLEXITY", "", 0),
    ("BEST_MATCH", "", 0),

    ("SCALA_BINARY", "", "scala"),
    ("SCALA_OPTS", "", "-J-Xmx3024M"),    

    ("COMMONS_MATH3_JAR", "", "/usr/share/java/commons-math3.jar"),
    )

def print_cmd_line(s, target, source, env):
    if len(s) > int(env["OUTPUT_WIDTH"]):
        print s[:int(env["OUTPUT_WIDTH"]) - 10] + "..." + s[-7:]
    else:
        print s

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

env = Environment(variables=vars, ENV=os.environ, TARFLAGS="-c -z", TARSUFFIX=".tgz",
                  tools=["default", "textfile"] + [x.TOOLS_ADD for x in [bhmm_tools]],
                  BUILDERS={"CopyFile" : Builder(action="cp ${SOURCE} ${TARGET}"),
                            },
                  )

if not os.path.exists(env.subst("${MORPHOLOGY_PATH}")):
    logging.error(env.subst("the variable MORPHOLOGY_PATH does not point to a valid location (${MORPHOLOGY_PATH})"))
    env.Exit()

env['PRINT_CMD_LINE_FUNC'] = print_cmd_line
env.Decider("MD5-timestamp")

data_sets = {}

for language_name in ["turkish"]: #["assamese", "bengali", "pashto", "tagalog", "turkish", "zulu"]:
    if os.path.exists("data/%s_morphology.txt" % language_name):
        x = env.StandardizedToBHMM("work/sentences/%s_limited.xml.gz" % language_name, ["data/transcripts/%s/train.ref" % language_name, "data/%s_morphology.txt" % language_name])
                                   #["${LANGUAGE_PACKS}/%d.tgz" % language_id, "data/%s_morphology.txt" % language_name])
    else:
        x = env.StandardizedToBHMM("work/sentences/%s_limited.xml.gz" % language_name, "data/transcripts/%s/train.ref" % language_name)
    data_sets[language_name] = x
        #x = env.StandardizedToBHMM("work/sentences/%s_limited.xml.gz" % language_name, "${LANGUAGE_PACKS}/%d.tgz" % language_id)
#    data_sets["%s_limited" % language] = env.StandardizedToBHMM("work/sentences/%s_limited.xml.gz" % language, "data/transcripts/%s/train.ref" % language)
#    data_sets["%s_dev" % language] = env.StandardizedToBHMM("work/sentences/%s_dev.xml.gz" % language, "data/transcripts/%s/dev.ref" % language)

#for language_id, language_name in env["LANGUAGES"].iteritems():
#    if os.path.exists("data/%s_morphology.txt" % language_name):
#        x = env.BabelToBHMM("work/sentences/%s.xml.gz" % language_name, ["${LANGUAGE_PACKS}/%d.tgz" % language_id, "data/%s_morphology.txt" % language_name])
#    else:
#        x = env.BabelToBHMM("work/sentences/%s.xml.gz" % language_name, "${LANGUAGE_PACKS}/%d.tgz" % language_id)
#     data_sets[language_name] = x

data_sets["zulu"] = env.UkwabelanaToBHMM(["work/sentences/ukwabelana_zulu.xml.gz"], "data/UkwabelanaCorpus.tar.gz")
full_english = env.PennToBHMM("work/sentences/english.xml.gz", ["data/penn_wsj_tag_data.txt.gz", "data/english_morphology.txt"])
data_sets["english"] = env.CreateSubset("work/sentences/english_subset.xml.gz", [full_english, Value(3000), Value(None)])

# data_sets["persian"] = env.UpcToBHMM("work/sentences/persian.xml.gz", "data/UPC.txt.tar.gz")
# data_sets["german"] = env.TigerToBHMM("work/sentences/german.xml.gz", "data/tigercorpus-2.2.xml.tar.gz")

java_classes = env.Java(pjoin("work", "classes"), pjoin(env["MORPHOLOGY_PATH"], "src"))

scala_classes = env.Scala(env.Dir(pjoin("work", "classes", "bhmm")), [env.Dir("src")] + java_classes)
env.Depends(scala_classes, java_classes)

mt = env.Jar(pjoin("work", "morphological_tagger.jar"), "work/classes", JARCHDIR="work/classes")

arguments = [("--%s" % x, "${%s}" % x.replace("-", "_").upper()) for x in ["markov", "num-sentences", "num-tags", "num-burnins", "num-samples", "transition-prior", "emission-prior", "variation-of-information", "perplexity", "best-match", "optimize-every", "annealing", "token-based"]] + [("--%s" % x, "") for x in ["symmetric-emission-prior", "symmetric-transition-prior"] if env[x.replace("-", "_").upper()]]

results = {}
for language, data in [x for x in data_sets.iteritems() if x[0].startswith("uk") or True]:
    has_gold_morph = (language in ["english", "zulu", "turkish"])
    has_gold_tags = False
    limited = language in env["LANGUAGES"].values()

    if has_gold_morph:
        gold_morph = env.DatasetToEmma("work/gold_standards/%s_morph.txt" % language, data)
        #env.EvaluateMorphology("work/gold_standards/%s_morph_dummy_evaluation.txt" % language, [gold_morph, gold_morph])        
        random = env.RandomSegmentations("work/morfessor/%s_random.xml.gz" % language, data)
        random_pred = env.DatasetToEmma("work/morfessor/%s_random_predictions.txt" % language, random)
        results[("random", language)] = env.EvaluateMorphology("work/morfessor/%s_morph_random_evaluation.txt" % language, [gold_morph, random_pred])

    #if has_gold_tags:
    #    env.EvaluateTagging("work/gold_standards/%s_tag_dummy_evaluation.txt" % language, [data, data])
    
    #
    # Morfessor experiments
    #
    output = env.TrainMorfessor("work/morfessor/%s.xml.gz" % (language), data, LIMITED=limited)
    tri_output = env.MorfessorToTripartite("work/morfessor/%s_tripartite.xml.gz" % language, output)
    if has_gold_morph:
        pred = env.DatasetToEmma("work/morfessor/%s_predictions.txt" % language, output)
        tri_pred = env.DatasetToEmma("work/morfessor/%s_tripartite_predictions.txt" % language, tri_output)        
        #env.EvaluateMorphology("work/morfessor/%s_morph_dummy_evaluation.txt" % language, [pred, pred])
        results[("morfessor", language)] = env.EvaluateMorphology("work/morfessor/%s_morph_evaluation.txt" % language, [gold_morph, pred])
        results[("tri_morfessor", language)] = env.EvaluateMorphology("work/morfessor/%s_morph_tripartite_evaluation.txt" % language, [gold_morph, tri_pred])

    #
    # Tagging experiments
    #
    output = env.RunScala("work/tagging/%s.xml.gz" % language, [mt, env.Value("bhmm.Main"), data],
                           ARGUMENTS=" ".join(sum(map(list, arguments + [("--mode", "tagging")]), [])), LIMITED=limited)
    env.Depends(output, mt)
    if has_gold_tags:
        env.EvaluateTagging("work/tagging/%s_tag_dummy_evaluation.txt" % language, [pred, pred])
        env.EvaluateTagging("work/tagging/%s_tag_evaluation.txt" % language, [data, pred])
    
    #
    # Morphology experiments
    #
    output = env.RunScala("work/morphology/%s.xml.gz" % language, [mt, env.Value("bhmm.Main"), data],
                           ARGUMENTS=" ".join(sum(map(list, arguments + [("--mode", "morphology")]), [])), LIMITED=limited)
    env.Depends(output, mt)
    env.XMLToSadegh(output)
    if has_gold_morph:
        pred = env.DatasetToEmma("work/morphology/%s_predictions.txt" % language, output)
        #env.EvaluateMorphology("work/morphology/%s_morph_dummy_evaluation.txt" % language, [pred, pred])
        results[("ag_morphology", language)] = env.EvaluateMorphology("work/morphology/%s_morph_evaluation.txt" % language, [gold_morph, pred])

    #
    # Joint experiments
    #
    output = env.RunScala("work/joint/%s.xml.gz" % language, [mt, env.Value("bhmm.Main"), data],
                           ARGUMENTS=" ".join(sum(map(list, arguments + [("--mode", "morphology")]), [])), LIMITED=limited)
    env.Depends(output, mt)
    # if has_gold_morph:
    #     pred = env.DatasetToEmma("work/joint/%s_predictions.txt" % language, output)
    #     env.EvaluateMorphology("work/joint/%s_morph_dummy_evaluation.txt" % language, [pred, pred])
    #     env.EvaluateMorphology("work/joint/%s_morph_evaluation.txt" % language, [gold_morph, pred])
    # if has_gold_tags:
    #     env.EvaluateTagging("work/joint/%s_tag_dummy_evaluation.txt" % language, [pred, pred])
    #     env.EvaluateTagging("work/joint/%s_tag_evaluation.txt" % language, [data, pred])

res = env.CollateResults("work/results.txt", results.values(), RESULTS=results)
