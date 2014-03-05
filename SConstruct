import os
import os.path
import sys
from SCons.Tool import textfile
from glob import glob
import logging
import time
from os.path import join as pjoin
from os import listdir
import re

vars = Variables("custom.py")
vars.AddVariables(
    ("OUTPUT_WIDTH", "", 130),
    BoolVariable("DEBUG", "", False),
    BoolVariable("TEST", "", False),
    ("MORPHOLOGY_PATH", "", False),
    ("NUM_SENTENCES", "", 10),
    ("NUM_TAGS", "", 17),
    ("NUM_BURNINS", "", 10),
    ("NUM_SAMPLES", "", 1),
    ("MARKOV", "", 2),

    ("TRANSITION_PRIOR", "", .1),
    ("EMISSION_PRIOR", "", .1),
    EnumVariable("OPTIMIZE_TRANSITION_PRIOR", "", "no", ["no", "symmetric", "asymmetric"]),
    EnumVariable("OPTIMIZE_EMISSION_PRIOR", "", "no", ["no", "symmetric", "asymmetric"]),
    ("OPTIMIZE_TRANSITION_PRIOR_EVERY", "", 0),
    ("OPTIMIZE_TRANSITION_PRIOR_EVERY", "", 0),

    ("VARIATION_OF_INFORMATION", "", 0),
    ("PERPLEXITY", "", 0),
    ("BEST_MATCH", "", 0),

    ("SCALA_OPTS", "", ""),    
    )

def print_cmd_line(s, target, source, env):
    if len(s) > int(env["OUTPUT_WIDTH"]):
        print s[:int(env["OUTPUT_WIDTH"]) - 10] + "..." + s[-7:]
    else:
        print s

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

def recursive_find(path, pattern):
    entries = [os.path.join(path, e) for e in listdir(path)]
    return [e for e in entries if os.path.isfile(e) and re.match(pattern, os.path.basename(e))] + sum([recursive_find(e, pattern) for e in entries if os.path.isdir(e)], [])

def scala_compile_generator(target, source, env, for_signature):
    return "scalac -cp work/classes:/usr/share/java/commons-math3.jar -d work/classes ${SOURCES}"

def scala_compile_emitter(target, source, env):
    new_sources = recursive_find(source[0].rstr(), r".*\.scala$")
    new_targets = [re.sub(r"^[^\/]*", target[0].rstr(), x.replace(".scala", ".class")) for x in new_sources]
    return new_targets, new_sources

env = Environment(variables=vars, ENV=os.environ, TARFLAGS="-c -z", TARSUFFIX=".tgz",
                  tools=["default", "textfile"] + [x.TOOLS_ADD for x in []],
                  BUILDERS={"Swig" : Builder(action="${SWIG_PATH} -o ${TARGETS[0]} -outdir ${SWIGOUTDIR} ${_CPPDEFFLAGS} ${SWIGFLAGS} ${SOURCES[0]}"),
                            "CopyFile" : Builder(action="cp ${SOURCE} ${TARGET}"),
                            "Scala" : Builder(generator=scala_compile_generator, emitter=scala_compile_emitter),
                            "InterpretScala" : Builder(action="scala ${SCALA_OPTS} ${SOURCES[0]} ${ARGUMENTS}"),
                            "RunScala" : Builder(action="scala -cp ${SOURCES[0]} ${SOURCES[1]} --input ${SOURCES[2]} --output ${TARGET} ${ARGUMENTS}"),
                            },
                  )

if not os.path.exists(env.subst("${MORPHOLOGY_PATH}")):
    logging.error(env.subst("the variable MORPHOLOGY_PATH does not point to a valid location (${MORPHOLOGY_PATH})"))
    env.Exit()

env['PRINT_CMD_LINE_FUNC'] = print_cmd_line
env.Decider("MD5-timestamp")

data = env.File("data/penn_wsj_tag_data.txt.gz")

java_classes = env.Java(pjoin("work", "classes"), pjoin(env["MORPHOLOGY_PATH"], "src"))

scala_classes = env.Scala(env.Dir(pjoin("work", "classes", "bhmm")), [env.Dir("src")] + java_classes)
env.Depends(scala_classes, java_classes)

mt = env.Jar(pjoin("work", "morphological_tagger.jar"), "work/classes", JARCHDIR="work/classes")

arguments = [("--%s" % x, "${%s}" % x.replace("-", "_").upper()) for x in ["markov", "num-sentences", "num-tags", "num-burnins", "num-samples", "transition-prior", "emission-prior", "optimize-transition-prior", "optimize-emission-prior", "variation-of-information", "perplexity", "best-match"]]

if env["TEST"]:
   arguments.append(["--test"])

output = env.RunScala("work/output.txt", [mt, env.Value("bhmm.Main"), data],
                      ARGUMENTS=" ".join(sum(map(list, arguments), [])))
env.Depends(output, mt)


#env.InterpretScala("work/output.txt", [pjoin("src", "BHMM.scala"), morphology, data], 
#             SCALA_OPTS="-cp ${SOURCES[1]}:/usr/share/java/commons-math3.jar",
#                      ARGUMENTS=" ".join(sum(map(list, arguments), [])))
#             

