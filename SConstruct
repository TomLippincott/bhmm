import os
import os.path
import sys
from SCons.Tool import textfile
from glob import glob
import logging
import time
from os.path import join as pjoin

vars = Variables("custom.py")
vars.AddVariables(
    ("OUTPUT_WIDTH", "", 130),
    BoolVariable("DEBUG", "", False),
    ("MORPHOLOGY_PATH", "", "/home/tom/projects/jointmorphologytagger"),
    ("NUM_SENTENCES", "", 10),
    ("NUM_TAGS", "", 17),
    ("NUM_BURNINS", "", 10),
    ("NUM_SAMPLES", "", 1),
    ("MARKOV", "", 2),
    ("TRANSITION_PRIOR", "", .1),
    ("EMISSION_PRIOR", "", .1),
    ("SCALA_OPTS", "", ""),
    )

def print_cmd_line(s, target, source, env):
    if len(s) > int(env["OUTPUT_WIDTH"]):
        print s[:int(env["OUTPUT_WIDTH"]) - 10] + "..." + s[-7:]
    else:
        print s

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

env = Environment(variables=vars, ENV=os.environ, TARFLAGS="-c -z", TARSUFFIX=".tgz",
                  tools=["default", "textfile"] + [x.TOOLS_ADD for x in []],
                  BUILDERS={"Swig" : Builder(action="${SWIG_PATH} -o ${TARGETS[0]} -outdir ${SWIGOUTDIR} ${_CPPDEFFLAGS} ${SWIGFLAGS} ${SOURCES[0]}"),
                            "CopyFile" : Builder(action="cp ${SOURCE} ${TARGET}"),
                            "Scala" : Builder(action="scalac ${SCALA_OPTS} ${SOURCES[1:]}"),
                            "RunScala" : Builder(action="scala ${SCALA_OPTS} ${SOURCES[0]} ${ARGUMENTS}"),
                            },
                  )

env['PRINT_CMD_LINE_FUNC'] = print_cmd_line
env.Decider("MD5-timestamp")

data = env.File("data/penn_wsj_tag_data.txt.gz")

java_classes = env.Java(pjoin("work", "classes"), pjoin(env["MORPHOLOGY_PATH"], "src"))

morphology = env.Jar(pjoin("work", "morphology.jar"), java_classes)

config = env.File(pjoin("data", "setting.txt"))

#env.Scala(source=[morphology] + env.Glob("src/*scala"),
#          SCALA_OPTS="-cp ${SOURCES[0]}")

env.RunScala("work/output.txt", [pjoin("src", "BHMM.scala"), morphology, config, data], 
             SCALA_OPTS="-cp ${SOURCES[1]}",
             ARGUMENTS="--perplexity --config ${SOURCES[2]} --output ${TARGET} --markov ${MARKOV} --num-sentences ${NUM_SENTENCES} --num-tags ${NUM_TAGS} --num-burnins ${NUM_BURNINS} --num-samples ${NUM_SAMPLES} --transition-prior ${TRANSITION_PRIOR} --emission-prior ${EMISSION_PRIOR} --input ${SOURCES[3]}")
