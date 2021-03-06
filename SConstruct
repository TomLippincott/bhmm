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
import pycfg_tools

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
    ("LANGUAGES", "", {}),
    BoolVariable("DEBUG", "", True),
    BoolVariable("HAS_TORQUE", "", False),
    ("ANNEAL_INITIAL", "", 5),
    ("ANNEAL_FINAL", "", 1),
    ("ANNEAL_ITERATIONS", "", 0),

    ("INDUSDB_PATH", "", None),

    ("CABAL", "", "/home/tom/.cabal"),
    ("PYCFG_PATH", "", ""),
    ("MAXIMUM_SENTENCE_LENGTH", "", 20),
    ("HASKELL_PATH", "", ""),
    
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
                                                                         trmorph_tools, sfst_tools, mila_tools, almor_tools, pycfg_tools]],
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
common_parameters = "--log-file ${TARGETS[1]} --num-burnins ${NUM_BURNINS} --num-samples ${NUM_SAMPLES} --mode ${MODE} ${TOKEN_BASED==True and '--token-based' or ''} --save-every ${SAVE_EVERY} ${LOWER_CASE_TAGGING==True and '--lower-case-tagging' or ''} ${LOWER_CASE_MORPHOLOGY==True and '--lower-case-morphology' or ''}"
tagging_parameters = " ".join(["--%s ${%s}" % (x.lower().replace("_", "-"), x) for x in ["NUM_TAGS", "MARKOV", "TRANSITION_PRIOR", "EMISSION_PRIOR"]] +
                              ["${%s and '--%s' or ''}" % (x, x.lower().replace("_", "-")) for x in ["SYMMETRIC_TRANSITION_PRIOR", "SYMMETRIC_EMISSION_PRIOR"]])
morphology_parameters = " ".join(["--%s ${%s}" % (x.lower().replace("_", "-"), x) for x in ["PREFIX_PRIOR", "SUFFIX_PRIOR", "SUBMORPH_PRIOR", "WORD_PRIOR", "TAG_PRIOR", "BASE_PRIOR", "ADAPTOR_PRIOR_A", "ADAPTOR_PRIOR_B", "CACHE_PROBABILITY", "RULE_PRIOR"]] +
                                 ["${%s==True and '--%s' or ''}" % (x, x.lower().replace("_", "-")) for x in ["MULTIPLE_STEMS", "SUFFIXES", "PREFIXES", "SUBMORPHS", "NON_PARAMETRIC", "HIERARCHICAL", "USE_HEURISTICS", "DERIVATIONAL", "INFER_PYP"]])

# compile the adaptor grammar and BHMM code into a single jar file
#java_classes = env.Java(pjoin("work", "classes"), pjoin(env["MORPHOLOGY_PATH"], "src"))
#scala_classes = env.Scala(env.Dir(pjoin("work", "classes", "bhmm")), [env.Dir("src")] + java_classes)
#env.Depends(scala_classes, java_classes)
#mt = env.Jar(pjoin("work", "morphological_tagger.jar"), "work/classes", JARCHDIR="work/classes")

# iterate over each language
data_sets = {}
tagging_results = {}
morphology_results = {}
oov_eval_quality = {}


#arguments = {"num_syntactic_classes" : 1,
#             "num_tags" : 10,
#             "markov" : 1,
#             }

arguments = {}

models = ["Tagging", 
          "Morphology", 
          "Joint", 
          "GoldTagsJoint", 
          "GoldSegmentationsJoint",
      ]

#for language, (lower_case_tagging, lower_case_morphology) in env["LANGUAGES"].iteritems():
for language, properties in env["LANGUAGES"].iteritems():
    env.Replace(LANGUAGE=language)
    babel_id = properties.get("BABEL_ID", None)
            #ngram_cfg, ngram_data = env.NGram(["work/models/${LANGUAGE}_ngram.txt", "work/data/${LANGUAGE}_ngram.txt.gz"], [data, Value({"lowercase" : True})])
            #parses, grammar, trace_file = env.RunPYCFG(["work/pycfg/${LANGUAGE}/ngram_%s.txt" % x for x in ["parses", "grammar", "trace"]], [ngram_cfg, ngram_data])
            #print language, babel_id, rtm
    #continue
    env.Replace(LANGUAGE=language)
    #env.Replace(LOWER_CASE_TAGGING=lower_case_tagging)
    #env.Replace(LOWER_CASE_MORPHOLOGY=lower_case_morphology)
    #arguments["LOWER_CASE_TAGGING"] = lower_case_tagging
    #arguments["LOWER_CASE_MORPHOLOGY"] = lower_case_morphology

    # data set
    if os.path.exists(env.subst("${EMNLP_DATA_PATH}/${LANGUAGE}/pos_cap/train.pos")):
        data = env.subst("${EMNLP_DATA_PATH}/${LANGUAGE}/pos_cap/train.pos")
    else:
        data = env.subst("${EMNLP_DATA_PATH}/${LANGUAGE}/pos/train.pos")

    # convert training data to XML format
    training = env.CONLLishToXML("work/data/${LANGUAGE}/train.xml.gz", data)

    # if we have gold standard morphology, add it in
    has_morphology = os.path.exists(env.subst("data/${LANGUAGE}_morphology.txt"))
    if has_morphology:
        training = env.AddMorphology("work/data/${LANGUAGE}/train_morph.xml.gz", [training, "data/${LANGUAGE}_morphology.txt"])
        #gold_morphology = env.DatasetToEMMA("work/pycfg/${LANGUAGE}/${MODEL}_morph_gold.txt", training)
    
        #if babel_id or True:
        #rtms = env.Glob("${INDUSDB_PATH}/IARPA-babel*/IARPA-babel%.3d*-dev.stm" % babel_id)
        #if len(rtms) == 1:
            #rtm = rtms[0]
            #data = env.RtmToData("work/data/${LANGUAGE}.txt.gz", rtm)
        #     ndata
    #prefixes, suffixes = env.ExtractAffixes(["work/affixes/${LANGUAGE}_prefixes.txt", "work/affixes/${LANGUAGE}_suffixes.txt"], training)
    #for keep in [0]: #, 50, 100, 500]:
    #    env.Replace(KEEP=keep)
    for has_prefixes in [True, False]:
        for has_suffixes in [True, False]:
            if has_prefixes == has_suffixes and not has_prefixes:
                continue
            arguments = Value({"MODEL" : "morphology",
                          "LANGUAGE" : language,
                          "HAS_PREFIXES" : has_prefixes,
                          "HAS_SUFFIXES" : has_suffixes,
                      })
            pycfg_model, pycfg_data = env.CreateModel([training, arguments])

            #pycfg_model, pycfg_data = env.CreateModel(["work/models/${LANGUAGE}_${HAS_PREFIXES}_${HAS_SUFFIXES}.txt", "work/models/${LANGUAGE}_${HAS_PREFIXES}_${HAS_SUFFIXES}_data.txt"], 
            #                                          training)
            parses, grammar, trace_file = env.RunPYCFG([pycfg_model, pycfg_data, arguments])
            if has_morphology:
                
                results = getattr(env, "EvaluateMorphology")(parses, training, "data/${LANGUAGE}_morphology.txt")

    continue


    # run all the different models
    for model in models:        
        env.Replace(MODEL=model)
        cfg, data = getattr(env, "%sCFG" % model)(["work/pycfg/${LANGUAGE}/${MODEL}.txt", "work/pycfg/${LANGUAGE}/${MODEL}_data.txt.gz"], [training, Value(arguments)])
        parses, grammar, trace_file = env.RunPYCFG(["work/pycfg/${LANGUAGE}/${MODEL}_%s.txt" % x for x in ["parses", "grammar", "trace"]], [cfg, data])
        #if has_morphology:
        results = getattr(env, "Evaluate%s" % model)("work/results/${LANGUAGE}_${MODEL}.txt", [parses, training])

    #
    # Morfessor experiments
    #
    morfessor_segmentations = env.TrainMorfessor("work/xml_formatted/${LANGUAGE}_morfessor.xml.gz", training)
    if has_morphology:
        env.Replace(MODEL="Morfessor")
        results = env.EMMAScore("work/results/${LANGUAGE}_${MODEL}.txt", morfessor_segmentations, training)

    continue
    # # OOV reduction evaluation data
    # small_oov = env.File("${EMNLP_DATA_PATH}/%s/oov_eval/small_eval.freq" % (language))
    # big_oov = env.File("${EMNLP_DATA_PATH}/%s/oov_eval/big_eval.freq" % (language))
    
    # # language acceptor
    # acceptor = env.get_builder("%sFilter" % (language.title()))
    
    # # run acceptors on training data
    # acceptor_quality = acceptor(env, ["work/acceptor_quality/%s_%s.txt" % (language, x) for x in ["good", "bad"]], training)

    # # for each expansion method
    # reductions = {}
    # for generation_method in ["joint", "adaptor", "iadaptor", "bbg"]:
    #     continue
    #     # data file from Sadegh
    #     expansion = env.Glob("${EMNLP_EXPERIMENTS_PATH}/%s/expansion/%s.*.gz" % (language, generation_method))[0] #, generation_method))[0]

    #     # filter based on acceptor
    #     accepted, rejected = acceptor(env, ["work/expansions/${LANGUAGE}/%s/%s.txt" % (generation_method, x) for x in ["accepted", "rejected"]], expansion, LANGUAGE=language)

    #     # split into different reranking methods
    #     expansions = {k : v for k, v in zip(["nrr", "tri", "tri_bound"],
    #                                         env.SplitExpansions(["work/expansions/${LANGUAGE}/%s/%s.txt" % (generation_method, x) for x in ["nrr", "tri", "tri_bound"]], 
    #                                                             [expansion, Value(100000)], LANGUAGE=language))}

    #     # create files tracking performance across N bins (token-based recall from gigaword, type-based precision from acceptor)
    #     for rank_method, expansion_file in expansions.iteritems():
    #         for size, oov in zip(["big", "small"], [big_oov, small_oov]):
    #             if size == "big" and rank_method == "tri":
    #                 #accepted, rejected = acceptor(env, ["work/expansions/${LANGUAGE}/%s/%s.txt" % (generation_method, x) for x in ["accepted", "rejected"]], expansion_file, LANGUAGE=language)
    #                 reductions[(generation_method, rank_method, size)] = env.OOVReduction("work/oov_reductions/%s/%s/%s_%s.txt" % (language, generation_method, rank_method, size), 
    #                                                                                       [training, expansion_file, oov, accepted, env.Value(1000)])
                
    # # plot reductions
    # env.PlotReduction("work/plots/${LANGUAGE}.png", reductions.values(), REDUCTIONS=reductions, LANGUAGE=language)


    for token_based in env["TOKEN_BASED"]:

        style_name = (token_based and "token-based") or "type-based"

        #
        # Random experiment
        #
        random_tags = env.RandomTags("work/xml_formatted/%s_random-tagging_%s.xml.gz" % (language, style_name), [training, env.Value(style_name)])
        random_segmentations = env.RandomSegmentations("work/xml_formatted/%s_random-morphology_%s.xml.gz" % (language, style_name), [training, env.Value(style_name)])
        tagging_results[(language, "random", style_name)] = env.EvaluateTagging("work/results/%s_random-tagging_%s.txt" % (language, style_name), [training, random_tags])
        morphology_results[(language, "random", style_name)] = env.EMMAScore("work/results/%s_random-morphology_%s.txt" % (language, style_name), training, random_segmentations[0])
        
        #
        # Tagging experiment
        #
        tagging, tagging_log = env.RunScala(["work/xml_formatted/%s_bhmm-tagging_%s.xml.gz" % (language, style_name), "work/logs/%s_bhmm-tagging_%s.txt" % (language, style_name)], 
                                            [mt, env.Value("bhmm.Main"), training],
                                            ARGUMENTS="%s %s" % (common_parameters, tagging_parameters), MODE="tagging", TOKEN_BASED=token_based, LOWER_CASE_TAGGING=lower_case_tagging)    
        tagging_results[(language, "tagger", style_name)] = env.EvaluateTagging("work/results/%s_bhmm-tagging_%s.txt" % (language, style_name), [training, tagging])
        env.TopWordsByTag("work/top_words/%s_bhmm-tagging_%s.txt" % (language, style_name), tagging)
        
        #
        # Morphology experiments
        #
        morphology, morphology_log = env.RunScala(["work/xml_formatted/%s_ag-morphology_%s.xml.gz" % (language, style_name), "work/logs/%s_ag-morphology_%s.txt" % (language, style_name)], 
                                                  [mt, env.Value("bhmm.Main"), training],
                                                  ARGUMENTS="%s %s" % (common_parameters, morphology_parameters), MODE="morphology", TOKEN_BASED=token_based, LOWER_CASE_MORPHOLOGY=lower_case_morphology)        
        morphology_results[(language, "morph", style_name)] = env.EMMAScore("work/results/%s_ag-morphology_%s.txt" % (language, style_name), training, morphology)
        continue
        #
        # Joint experiments
        #
        joint, joint_log = env.RunScala(["work/xml_formatted/%s_joint-model_%s.xml.gz" % (language, style_name), "work/logs/%s_joint-model_%s.log" % (language, style_name)], 
                           [mt, env.Value("bhmm.Main"), training],
                           ARGUMENTS="%s %s %s" % (common_parameters, tagging_parameters, morphology_parameters), MODE="joint", TOKEN_BASED=token_based, LOWER_CASE_TAGGING=lower_case_tagging, LOWER_CASE_MORPHOLOGY=lower_case_morphology)
        tagging_results[(language, "joint", style_name)] = env.EvaluateTagging("work/results/%s_joint-tagging_%s.txt" % (language, style_name), [training, joint])
        morphology_results[(language, "joint", style_name)] = env.EMMAScore("work/results/%s_joint-morphology_%s.txt" % (language, style_name), training, joint)

#env.CollateResults("work/results/summary.txt", morphology_results.values() + tagging_results.values(), MORPHOLOGY_RESULTS=morphology_results, TAGGING_RESULTS=tagging_results)
