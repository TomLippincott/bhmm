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
import emma_tools
import trmorph_tools
import sfst_tools
import evaluation_tools
import almor_tools
import mila_tools

vars = Variables("custom.py")
vars.AddVariables(
    ("OUTPUT_WIDTH", "", 130),
    ("SCALA_BINARY", "", "scala"),
    ("SCALA_OPTS", "", "-J-Xmx3024M"),    
    ("MORPHOLOGY_PATH", "", False),
    ("EMNLP_DATA_PATH", "", False),
    ("EMNLP_EXPERIMENTS_PATH", "", False),
    ("EMNLP_TOOLS_PATH", "", False),
    ("LOCAL_PATH", "", False),
    ("LANGUAGES", "", []),
    
    # parameters shared by all models
    ("NUM_BURNINS", "", 1),
    ("NUM_SAMPLES", "", 1),
    ("SAVE_EVERY", "", 1),
    ("TOKEN_BASED", "", [True, False]),

    # tagging parameters
    ("NUM_TAGS", "", 45),
    ("MARKOV", "", 1),
    ("TRANSITION_PRIOR", "", .1),
    ("EMISSION_PRIOR", "", .1),
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
    BoolVariable("MULTIPLE_STEMS", "", True),
    BoolVariable("PREFIXES", "", True),
    BoolVariable("SUFFIXES", "", True),
    BoolVariable("SUBMORPHS", "", True),
    BoolVariable("NON_PARAMETRIC", "", False),
    BoolVariable("HIERARCHICAL", "", True),
    BoolVariable("USE_HEURISTICS", "", False),
    BoolVariable("DERIVATIONAL", "", False),    
    BoolVariable("INFER_PYP", "", False),
    )

# initialize logging system
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

# initialize build environment
env = Environment(variables=vars, ENV=os.environ, TARFLAGS="-c -z", TARSUFFIX=".tgz",
                  tools=["default", "textfile"] + [x.TOOLS_ADD for x in [bhmm_tools, evaluation_tools, scala_tools, morfessor_tools, emma_tools, 
                                                                         trmorph_tools, sfst_tools, mila_tools, almor_tools]],
                  )

# MORPHOLOGY_PATH must point to checkout of adaptor-grammar code
if not env["MORPHOLOGY_PATH"] or not os.path.exists(env.subst("${MORPHOLOGY_PATH}")):
    logging.error(env.subst("the variable MORPHOLOGY_PATH does not point to a valid location (${MORPHOLOGY_PATH})"))
    env.Exit()

# don't print out lines longer than the terminal width
def print_cmd_line(s, target, source, env):
    if len(s) > int(env["OUTPUT_WIDTH"]):
        print s[:int(env["OUTPUT_WIDTH"]) - 10] + "..." + s[-7:]
    else:
        print s

env['PRINT_CMD_LINE_FUNC'] = print_cmd_line
env.Decider("MD5-timestamp")

# argument format for running each model (ugly, but allows one to use the jar file as a standalone)
common_parameters = "--log-file ${TARGETS[1]} --num-burnins ${NUM_BURNINS} --num-samples ${NUM_SAMPLES} --mode ${MODE} ${TOKEN_BASED==True and '--token-based' or ''} --save-every ${SAVE_EVERY}"
tagging_parameters = " ".join(["--%s ${%s}" % (x.lower().replace("_", "-"), x) for x in ["NUM_TAGS", "MARKOV", "TRANSITION_PRIOR", "EMISSION_PRIOR"]] +
                              ["${%s and '--%s' or ''}" % (x, x.lower().replace("_", "-")) for x in ["SYMMETRIC_TRANSITION_PRIOR", "SYMMETRIC_EMISSION_PRIOR"]])
morphology_parameters = " ".join(["--%s ${%s}" % (x.lower().replace("_", "-"), x) for x in ["PREFIX_PRIOR", "SUFFIX_PRIOR", "SUBMORPH_PRIOR", "WORD_PRIOR", "TAG_PRIOR", "BASE_PRIOR", "ADAPTOR_PRIOR_A", "ADAPTOR_PRIOR_B", "CACHE_PROBABILITY", "RULE_PRIOR"]] +
                                 ["${%s==True and '--%s' or ''}" % (x, x.lower().replace("_", "-")) for x in ["MULTIPLE_STEMS", "SUFFIXES", "PREFIXES", "SUBMORPHS", "NON_PARAMETRIC", "HIERARCHICAL", "USE_HEURISTICS", "DERIVATIONAL", "INFER_PYP"]])

# compile the adaptor grammar and BHMM code into a single jar file
java_classes = env.Java(pjoin("work", "classes"), pjoin(env["MORPHOLOGY_PATH"], "src"))
scala_classes = env.Scala(env.Dir(pjoin("work", "classes", "bhmm")), [env.Dir("src")] + java_classes)
env.Depends(scala_classes, java_classes)
mt = env.Jar(pjoin("work", "morphological_tagger.jar"), "work/classes", JARCHDIR="work/classes")

# iterate over each language
data_sets = {}
tagging_results = {}
morphology_results = {}
oov_eval_quality = {}

for language in env["LANGUAGES"]:
    
    # data set
    if os.path.exists(env.subst(os.path.join("${EMNLP_DATA_PATH}", language, "pos_cap", "train.pos"))):
        data = env.subst(os.path.join("${EMNLP_DATA_PATH}", language, "pos_cap", "train.pos"))
    else:
        data = env.subst(os.path.join("${EMNLP_DATA_PATH}", language, "pos", "train.pos"))

    # convert training data to XML format
    training = env.CONLLishToXML(os.path.join("work", "data", language, "train.xml.gz"), data)
    
    # if we have gold standard morphology, add it in
    if os.path.exists("data/%s_morphology.txt" % language):
        training = env.AddMorphology(os.path.join("work", "data", language, "train_morph.xml.gz"), [training, "data/%s_morphology.txt" % language])
    
    # OOV reduction evaluation data
    small_oov = env.File("${EMNLP_DATA_PATH}/%s/oov_eval/small_eval.freq" % (language))
    big_oov = env.File("${EMNLP_DATA_PATH}/%s/oov_eval/big_eval.freq" % (language))
    
    # language acceptor
    acceptor = env.get_builder("%sFilter" % (language.title()))
    
    # run acceptors on training data to get baseline acceptor quality
    #acceptor_quality = acceptor(env, ["work/acceptor_quality/%s_%s.txt" % (language, x) for x in ["good", "bad"]], training)

    # good, bad = acceptor(env, ["work/oov_word_acceptance/%s_%s_small_eval.txt.gz" % (type, language) for type in ["good", "bad"]], small_oov)
    # oov_eval_quality[(language, "small")] = [small_oov, good, bad]
    
    # good, bad = acceptor(env, ["work/oov_word_acceptance/%s_%s_big_eval.txt.gz" % (type, language) for type in ["good", "bad"]], big_oov)
    # oov_eval_quality[(language, "big")] = [big_oov, good, bad]
    
    # for each expansion method
    reductions = {}
    for generation_method in ["joint", "adaptor", "iadaptor", "bbg"]:
        continue
        # data file from Sadegh
        expansion = env.Glob("${EMNLP_EXPERIMENTS_PATH}/%s/expansion/%s.*.gz" % (language, generation_method))[0] #, generation_method))[0]

        # filter based on acceptor
        #accepted, rejected = acceptor(env, ["work/expansions/${LANGUAGE}/%s/%s.txt" % (generation_method, x) for x in ["accepted", "rejected"]], expansion, LANGUAGE=language)

        # split into different reranking methods
        expansions = {k : v for k, v in zip(["nrr", "tri", "tri_bound"],
                                            env.SplitExpansions(["work/expansions/${LANGUAGE}/%s/%s.txt" % (generation_method, x) for x in ["nrr", "tri", "tri_bound"]], 
                                                                [expansion, Value(100000)], LANGUAGE=language))}
        #continue        
        # create files tracking performance across N bins (token-based recall from gigaword, type-based precision from acceptor)
        for rank_method, expansion_file in expansions.iteritems():
            for size, oov in zip(["big", "small"], [big_oov, small_oov]):
                if size == "big" and rank_method == "tri":
                    accepted, rejected = acceptor(env, ["work/expansions/${LANGUAGE}/%s/%s.txt" % (generation_method, x) for x in ["accepted", "rejected"]], expansion_file, LANGUAGE=language)
                    reductions[(generation_method, rank_method, size)] = env.OOVReduction("work/oov_reductions/%s/%s/%s_%s.txt" % (language, generation_method, rank_method, size), 
                                                                                          [training, expansion_file, oov, accepted, env.Value(1000)])
                
    # plot reductions
    #env.PlotReduction("work/plots/${LANGUAGE}.png", reductions.values(), REDUCTIONS=reductions, LANGUAGE=language)

    #continue
    #
    # Morfessor experiments
    #
    #morfessor_segmentations = env.TrainMorfessor("work/xml_formatted/%s_morfessor.xml.gz" % (language), training)
    #morphology_results[(language, "morf", "CAT")] = env.EMMAScore("work/evaluations/%s_morph_morfessor.txt" % (language), training, morfessor_segmentations[0])

    for token_based in env["TOKEN_BASED"]:

        style_name = (token_based and "token-based") or "type-based"

        #
        # Random experiment
        #
        #random_segmentations = env.RandomSegmentations("work/xml_formatted/%s_%s_random_morph.xml.gz" % (language, style_name), [training, env.Value(style_name)])
        #random_tags = env.RandomTags("work/xml_formatted/%s_%s_random_tag.xml.gz" % (language, style_name), [training, env.Value(style_name)])
        #tagging_results[(language, "random", style_name)] = env.EvaluateTagging("work/results/%s_tagging_random_%s.txt" % (language, style_name), [training, random_tags])
        #morphology_results[(language, "random", style_name)] = env.EMMAScore("work/evaluations/%s_morph_random_%s.txt" % (language, style_name), training, random_segmentations[0])

        #
        # Tagging experiment
        #
        tagging, tagging_log = env.RunScala(["work/xml_formatted/%s_tagging-%s.xml.gz" % (language, style_name), "work/logs/%s_tagging-%s.txt" % (language, style_name)], 
                                            [mt, env.Value("bhmm.Main"), training],
                                            ARGUMENTS="%s %s" % (common_parameters, tagging_parameters), MODE="tagging", TOKEN_BASED=token_based)    
        tagging_results[(language, "tagger", style_name)] = env.EvaluateTagging("work/results/%s_tagging_%s.txt" % (language, style_name), [training, tagging])
        env.TopWordsByTag("work/top_words/%s_%s.txt" % (language, style_name), tagging)

        #
        # Morphology experiments
        #
        #morphology, morphology_log = env.RunScala(["work/xml_formatted/%s_ag-morph_%s.xml.gz" % (language, style_name), "work/logs/%s_%s_ag-morph.txt" % (language, style_name)], 
        #                                          [mt, env.Value("bhmm.Main"), training],
        #                                          ARGUMENTS="%s %s" % (common_parameters, morphology_parameters), MODE="morphology", TOKEN_BASED=token_based)        
        #morphology_results[(language, "morph", style_name)] = env.EMMAScore("work/results/%s_morphology_%s.txt" % (language, style_name), training, morphology)

        #
        # Joint experiments
        #
        #joint, joint_log = env.RunScala(["work/xml_formatted/%s_joint_%s.xml.gz" % (language, style_name), "work/logs/%s_joint-%s.log" % (language, style_name)], 
        #                     [mt, env.Value("bhmm.Main"), training],
        #                     ARGUMENTS="%s %s %s" % (common_parameters, tagging_parameters, morphology_parameters), MODE="joint", TOKEN_BASED=token_based)
        #tagging_results[(language, "joint", style_name)] = env.EvaluateTagging("work/results/%s_joint-tagging_%s.txt" % (language, style_name), [training, joint])
        #morphology_results[(language, "joint", style_name)] = env.EMMAScore("work/results/%s_joint-morphology_%s.txt" % (language, style_name), training, joint)
        #env.XMLToSadegh("work/for_sadegh/%s_%s.txt" % (language, style_name), joint)
#env.CollateResults("work/results/summary.txt", morphology_results.values() + tagging_results.values(), MORPHOLOGY_RESULTS=morphology_results, TAGGING_RESULTS=tagging_results)
#env.CollateOOVQuality("work/oov_word_acceptance/summary.txt", sum(oov_eval_quality.values(), []), FILES=oov_eval_quality)
