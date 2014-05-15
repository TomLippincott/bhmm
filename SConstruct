import os
import os.path
import sys
from SCons.Tool import textfile
from glob import glob
import logging
import time
import re
from os.path import join as pjoin
import bhmm_tools
import scala_tools
import morfessor_tools
import openfst_tools
import evaluation_tools

vars = Variables("custom.py")
vars.AddVariables(
    ("OUTPUT_WIDTH", "", 130),
    ("SCALA_BINARY", "", "scala"),
    ("SCALA_OPTS", "", "-J-Xmx3024M"),    
    ("MORPHOLOGY_PATH", "", False),
    ("EMNLP_BASE_PATH", "", False),

    # common parameters
    ("NUM_BURNINS", "", 1),
    ("NUM_SAMPLES", "", 1),

    # tagging parameters
    ("NUM_TAGS", "", 10),
    ("MARKOV", "", 2),
    ("TRANSITION_PRIOR", "", .01),
    ("EMISSION_PRIOR", "", .01),
    BoolVariable("SYMMETRIC_TRANSITION_PRIOR", "", True),
    BoolVariable("SYMMETRIC_EMISSION_PRIOR", "", True),

    # morphology parameters
    ("PREFIX_PRIOR", "", 1),
    ("SUFFIX_PRIOR", "", 1),
    ("SUBMORPH_PRIOR", "", 1),
    ("WORD_PRIOR", "", 1),
    ("TAG_PRIOR", "", .1),
    ("BASE_PRIOR", "", 1),    
    ("ADAPTOR_PRIOR_A", "", 0),
    ("ADAPTOR_PRIOR_B", "", 100),
    ("CACHE_PROBABILITY", "", 100),
    ("RULE_PRIOR", "", 1),
    #("LEX_GEN", "", 0),
    BoolVariable("MULTIPLE_STEMS", "", True),
    BoolVariable("PREFIXES", "", True),
    BoolVariable("SUFFIXES", "", True),
    BoolVariable("SUBMORPHS", "", True),
    BoolVariable("NON_PARAMETRIC", "", False),
    BoolVariable("HIERARCHICAL", "", True),
    BoolVariable("USE_HEURISTICS", "", False),
    BoolVariable("DERIVATIONAL", "", False),    
    BoolVariable("INFER_PYP", "", False),
    #BoolVariable("BATCH", "", True),
    )

common_parameters = "--log-file ${TARGETS[1]} --num-burnins ${NUM_BURNINS} --num-samples ${NUM_SAMPLES} --mode ${MODE} ${TOKEN_BASED==True and '--token-based' or ''}"
tagging_parameters = " ".join(["--%s ${%s}" % (x.lower().replace("_", "-"), x) for x in ["NUM_TAGS", "MARKOV", "TRANSITION_PRIOR", "EMISSION_PRIOR"]] +
                              ["${%s and '--%s' or ''}" % (x, x.lower().replace("_", "-")) for x in ["SYMMETRIC_TRANSITION_PRIOR", "SYMMETRIC_EMISSION_PRIOR"]])
morphology_parameters = " ".join(["--%s ${%s}" % (x.lower().replace("_", "-"), x) for x in ["PREFIX_PRIOR", "SUFFIX_PRIOR", "SUBMORPH_PRIOR", "WORD_PRIOR", "TAG_PRIOR", "BASE_PRIOR", "ADAPTOR_PRIOR_A", "ADAPTOR_PRIOR_B", "CACHE_PROBABILITY", "RULE_PRIOR"]] +
                                 ["${%s==True and '--%s' or ''}" % (x, x.lower().replace("_", "-")) for x in ["MULTIPLE_STEMS", "SUFFIXES", "PREFIXES", "SUBMORPHS", "NON_PARAMETRIC", "HIERARCHICAL", "USE_HEURISTICS", "DERIVATIONAL", "INFER_PYP"]])


def print_cmd_line(s, target, source, env):
    if len(s) > int(env["OUTPUT_WIDTH"]):
        print s[:int(env["OUTPUT_WIDTH"]) - 10] + "..." + s[-7:]
    else:
        print s

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

env = Environment(variables=vars, ENV=os.environ, TARFLAGS="-c -z", TARSUFFIX=".tgz",
                  tools=["default", "textfile"] + [x.TOOLS_ADD for x in [bhmm_tools, scala_tools, morfessor_tools, openfst_tools, evaluation_tools]],
                  BUILDERS={"CopyFile" : Builder(action="cp ${SOURCE} ${TARGET}"),
                            },
                  )

if not os.path.exists(env.subst("${MORPHOLOGY_PATH}")):
    logging.error(env.subst("the variable MORPHOLOGY_PATH does not point to a valid location (${MORPHOLOGY_PATH})"))
    env.Exit()

env['PRINT_CMD_LINE_FUNC'] = print_cmd_line
env.Decider("MD5-timestamp")

data_sets = {}
for language in ["arabic", "english", "german", "hebrew", "turkish"]:
    small_oov = env.File("${EMNLP_BASE_PATH}/emnlp2014_data/%s/oov_eval/small_eval.freq" % (language))
    big_oov = env.File("${EMNLP_BASE_PATH}/emnlp2014_data/%s/oov_eval/big_eval.freq" % (language))
    for mode in ["train", "test", "dev"]:
        data_sets[mode] = data_sets.get(mode, {})
        data_sets[mode][language] = env.CONLLishToXML(os.path.join("work", "data", language, "%s.xml.gz" % mode), 
                                                      os.path.join("${EMNLP_BASE_PATH}/emnlp2014_data", language, "pos", "%s.pos" % mode))

    for generation_method in ["adaptor", "iadaptor", "bbg"]:
        expansions = {k : v for k, v in zip(["nrr", "tri", "tri_bound"],
                                            env.SplitExpansions(["work/expansions/${LANGUAGE}/%s/%s.txt" % (generation_method, x) for x in ["nrr", "tri", "tri_bound"]], 
                                                                "${EMNLP_BASE_PATH}/emnlp2014_experiments/%s/expansion/%s.ex" % (language, generation_method), LANGUAGE=language))}
        files = {}
        for rank_method, file_name in expansions.iteritems():
            for size, oov in zip(["big", "small"], [big_oov, small_oov]):
                files[(language, generation_method, rank_method, size)] = env.OOVReduction("work/reduction/%s/%s/%s_%s.txt" % (language, generation_method, rank_method, size), [data_sets["train"][language], file_name, oov])
        env.PlotReduction("work/plots/${LANGUAGE}_${GENERATION_METHOD}.png", files.values(), FILES=files, LANGUAGE=language, GENERATION_METHOD=generation_method)



# for language_name in ["turkish"]: #"assamese", "bengali", "pashto", "tagalog", "turkish", "zulu"]:
#     continue
#     if os.path.exists("data/%s_morphology.txt" % language_name):
#         x = env.StandardizedToBHMM("work/sentences/%s_limited.xml.gz" % language_name, ["data/transcripts/%s/train.ref" % language_name, "data/%s_morphology.txt" % language_name])
#                                    #["${LANGUAGE_PACKS}/%d.tgz" % language_id, "data/%s_morphology.txt" % language_name])
#     else:
#         x = env.StandardizedToBHMM("work/sentences/%s_limited.xml.gz" % language_name, "data/transcripts/%s/train.ref" % language_name)
#     data_sets[language_name] = x
#         #x = env.StandardizedToBHMM("work/sentences/%s_limited.xml.gz" % language_name, "${LANGUAGE_PACKS}/%d.tgz" % language_id)
# #    data_sets["%s_limited" % language] = env.StandardizedToBHMM("work/sentences/%s_limited.xml.gz" % language, "data/transcripts/%s/train.ref" % language)
# #    data_sets["%s_dev" % language] = env.StandardizedToBHMM("work/sentences/%s_dev.xml.gz" % language, "data/transcripts/%s/dev.ref" % language)

# #for language_id, language_name in env["LANGUAGES"].iteritems():
# #    if os.path.exists("data/%s_morphology.txt" % language_name):
# #        x = env.BabelToBHMM("work/sentences/%s.xml.gz" % language_name, ["${LANGUAGE_PACKS}/%d.tgz" % language_id, "data/%s_morphology.txt" % language_name])
# #    else:
# #        x = env.BabelToBHMM("work/sentences/%s.xml.gz" % language_name, "${LANGUAGE_PACKS}/%d.tgz" % language_id)
# #     data_sets[language_name] = x

# data_sets["zulu"] = env.UkwabelanaToBHMM(["work/sentences/ukwabelana_zulu.xml.gz"], "data/UkwabelanaCorpus.tar.gz")
full_english = env.PennToBHMM("work/sentences/english.xml.gz", ["data/penn_wsj_tag_data.txt.gz", "data/english_morphology.txt"])
# #data_sets["english"] = env.CreateSubset("work/sentences/english_subset.xml.gz", [full_english, Value(1500), Value(None)])
# #data_sets["english"] = env.CreateSubset("work/sentences/tiny_english_subset.xml.gz", [full_english, Value(3), Value(None)])

# # data_sets["persian"] = env.UpcToBHMM("work/sentences/persian.xml.gz", "data/UPC.txt.tar.gz")
data_sets["german"] = env.TigerToBHMM("work/sentences/german.xml.gz", "data/tigercorpus-2.2.xml.tar.gz")

java_classes = env.Java(pjoin("work", "classes"), pjoin(env["MORPHOLOGY_PATH"], "src"))
scala_classes = env.Scala(env.Dir(pjoin("work", "classes", "bhmm")), [env.Dir("src")] + java_classes)
env.Depends(scala_classes, java_classes)

mt = env.Jar(pjoin("work", "morphological_tagger.jar"), "work/classes", JARCHDIR="work/classes")

morph_results = {}
tag_results = {}

for language, data in data_sets["train"].iteritems():
    has_gold_morph = False #(language in ["english", "zulu", "turkish"])
    has_gold_tags = False #(language in ["zulu"])
    limited = False
    #limited = language in env["LANGUAGES"].values()

    if has_gold_morph:
        gold_morph = env.DatasetToEmma("work/emma_formatted/%s_morph.txt" % (language), data)
        morph_results[("gold", "gold", language)] = env.EvaluateMorphology("work/evaluations/%s_morph_dummy_gold.txt" % (language), [gold_morph, gold_morph])        
        random = env.RandomSegmentations("work/predictions/%s_random.xml.gz" % (language), data)
        random_pred = env.DatasetToEmma("work/emma_formatted/%s_random.txt" % (language), random)
        morph_results[("random", "gold", language)] = env.EvaluateMorphology("work/evaluations/%s_morph_random.txt" % (language), [gold_morph, random_pred])

    #if has_gold_tags:
    #    tag_results[("gold", "gold", language)] = env.EvaluateTagging("work/evaluations/%s_tag_dummy.txt" % (language), [data, data])


    #
    # Morfessor experiments
    #
    output = env.TrainMorfessor("work/xml_formatted/%s_morfessor.xml.gz" % (language), data, LIMITED=limited)
    #env.Depends(output, mt)
    tri_output = env.MorfessorToTripartite("work/xml_formatted/%s_morfessor_tripartite.xml.gz" % language, output)
    # fst, fst_syms = env.SegmentationToFST(["work/output/morfessor/%s_tripartite.fst" % language, 
    #                                        "work/output/morfessor/%s_tripartite.syms" % language],
    #                                       tri_output)
    # words = env.GenerateWords("work/output/morfessor/%s_words.txt.gz" % language, [fst, fst_syms, Value(1000)])
    # reranked = env.RerankByNgrams("work/output/morfessor/%s_reranked_words.txt.gz" % language, [words, Value(3)])
    if has_gold_morph:
        pred = env.DatasetToEmma("work/emma_formatted/%s_morfessor.txt" % language, output)
        tri_pred = env.DatasetToEmma("work/emma_formatted/%s_morfessor_tripartite.txt" % language, tri_output)        
        morph_results[("morfessor", "morfessor", language)] = env.EvaluateMorphology("work/evaluations/%s_morph_morfessor_morfessor.txt" % language, [pred, pred])
        morph_results[("morfessor", "gold", language)] = env.EvaluateMorphology("work/evaluations/%s_morph_morfessor.txt" % language, [gold_morph, pred])
        morph_results[("tri_morfessor", "tri_morfessor", language)] = env.EvaluateMorphology("work/evaluations/%s_morph_trimorfessor_trimorfessor.txt" % language, [tri_pred, tri_pred])
        morph_results[("tri_morfessor", "gold", language)] = env.EvaluateMorphology("work/evaluations/%s_morph_morfessor_tripartite.txt" % language, [gold_morph, tri_pred])



    for token_based in [True]:

        if token_based:
            style = "token-based"
        else:
            style = "type-based"

        #
        # Tagging experiments
        #
        output = env.RunScala(["work/xml_formatted/%s_tagging-%s.xml.gz" % (language, style), "work/logs/%s_tagging-%s.txt" % (language, style)], [mt, env.Value("bhmm.Main"), data],
                              ARGUMENTS="%s %s" % (common_parameters, tagging_parameters), MODE="tagging", TOKEN_BASED=token_based)
        #env.Depends(output, mt)

        #env.TopWordsByTag("work/output/tagging/%s_top_words_by_tag.txt.gz" % language, output)
        #if has_gold_tags:
        #    #env.EvaluateTagging("work/evaluation/tagging/%s_tag_dummy_evaluation.txt" % language, [pred, pred])
            #tag_results[("ag-tagging-%s" % style, "gold", language)] = env.EvaluateTagging("work/evaluation/tagging/%s_tag_evaluation.txt" % language, [data, output])
            #tag_results[("ag-tagging-%s" % style, "ag-tagging-%s" % style, language)] = env.EvaluateTagging("work/evaluations/tagging/%s_tag_evaluation.txt" % language, [output, output])

        #
        # Morphology experiments
        #
        output = env.RunScala(["work/xml_formatted/%s_ag-morph_%s.xml.gz" % (language, style), "work/logs/%s_%s_ag-morph.txt" % (language, style)], [mt, env.Value("bhmm.Main"), data],
                              ARGUMENTS="%s %s" % (common_parameters, morphology_parameters), MODE="morphology", TOKEN_BASED=token_based)
        #env.Depends(output, mt)
        #env.XMLToSadegh("work/sadegh_formatted/%s_ag-morphology_analyses.txt" % language, output)
        if has_gold_morph:
            pred = env.DatasetToEmma("work/emma_formatted/%s_%s_ag-morph.txt" % (language, style), output)
            morph_results[("ag-morphology-%s" % style, "ag-morphology-%s" % style, language)] = env.EvaluateMorphology("work/morphology/%s_ag-morphology-%s_ag-morphology-%s.txt" % (language, style, style), [pred, pred])
            morph_results[("ag-morphology-%s" % style, "gold", language)] = env.EvaluateMorphology("work/evaluations/%s_gold_ag-morphology-%s.txt" % (language, style), [gold_morph, pred])

        #
        # Joint experiments
        #
        output = env.RunScala(["work/xml_formatted/%s_joint_%s.xml.gz" % (language, style), "work/logs/%s_joint-%s.log" % (language, style)], [mt, env.Value("bhmm.Main"), data],
                              ARGUMENTS="%s %s %s" % (common_parameters, tagging_parameters, morphology_parameters), MODE="joint", TOKEN_BASED=token_based)
        #env.Depends(output, mt)
        #env.TopWordsByTag("work/output/joint/%s_top_words_by_tag.txt.gz" % language, output)
        if has_gold_morph:
            pred = env.DatasetToEmma("work/emma_formatted/%s_joint-%s.txt" % (language, style), output)
            #env.EvaluateMorphology("work/joint/%s_morph_dummy_evaluation.txt" % language, [pred, pred])
            morph_results[("ag_joint_morphology-%s" % style, "gold", language)] = env.EvaluateMorphology("work/evaluations/%s_gold_ag-joint-morphology-%s.txt" % (language, style), [gold_morph, pred])
        #if has_gold_tags:
        #    tag_results[("ag_joint_tagging", language)] = env.EvaluateTagging("work/evaluation/joint/%s_tag_evaluation.txt" % language, [data, output])
        #     env.EvaluateTagging("work/joint/%s_tag_dummy_evaluation.txt" % language, [pred, pred])
        #     env.EvaluateTagging("work/joint/%s_tag_evaluation.txt" % language, [data, pred])

#res = env.CollateResults("work/results.txt", morph_results.values() + tag_results.values(), MORPH_RESULTS=morph_results, TAG_RESULTS=tag_results)
