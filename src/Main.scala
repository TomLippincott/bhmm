package bhmm

import java.io.OutputStreamWriter
import java.util.zip._
import java.io.File
import java.io._
import java.util.HashMap
import java.util.Vector
import java.util.Date
import java.util.logging.{Level, Logger, Formatter, FileHandler, ConsoleHandler, LogRecord}
import bhmm.Test.{runTests}
import bhmm.data.DataSet
import bhmm.models.{TokenBasedTaggingModel, TypeBasedTaggingModel, TokenBasedMorphologyModel, TypeBasedMorphologyModel, TokenBasedJointModel, TypeBasedJointModel}
import bhmm.evaluation.Intrinsic
import AdaptorGrammar.PitmanYorPrior
import Distributions.{GammaDistribution, BetaDistribution}

object Main{  

  class MyFormatter extends Formatter{
    def format(r : LogRecord) : String = {
      r.getMessage + "\n"
      //val d = new Date(r.getMillis)
      //d.toString + ": " + r.getMessage + "\n"
    }
  }
  
  val usage = ""
  /*
  All arguments besides input and output are optional:
    --config
    --input
    --output
    --markov
    --num-sentences
    --num-tags
    --num-burnins
    --num-samples
    --transition-prior
    --emission-prior
    --help
  """
  */

  def main(args: Array[String]){
    type OptionMap = Map[Symbol, Any]
    if(args.length == 0) println(usage)
    val arglist = args.toList
    def nextOption(map : OptionMap, list: List[String]) : OptionMap = {
      def isSwitch(s : String) = (s(0) == '-')
      list match {
        case Nil => map

	// general options
        case "--input" :: value :: tail =>
          nextOption(map ++ Map('input -> value), tail)
        case "--output" :: value :: tail =>
          nextOption(map ++ Map('output -> value), tail)
        case "--log-file" :: value :: tail =>
          nextOption(map ++ Map('logFile -> value), tail)
	case "--mode" :: value :: tail =>
	  nextOption(map ++ Map('mode -> value), tail)
	case "--token-based" :: tail =>
	  nextOption(map ++ Map('typeBased -> false), tail)
	case "--num-burnins" :: value :: tail =>
	  nextOption(map ++ Map('numBurnins -> value.toInt), tail)
	case "--num-samples" :: value :: tail =>
	  nextOption(map ++ Map('numSamples -> value.toInt), tail)
	case "--save-every" :: value :: tail =>
	  nextOption(map ++ Map('saveEvery -> value.toInt), tail)

	// options for unsupervised tagger
	case "--markov" :: value :: tail =>
          nextOption(map ++ Map('markov -> value.toInt), tail)	  
	//case "--num-sentences" :: value :: tail =>
        //  nextOption(map ++ Map('numSentences -> value.toInt), tail)	  
	case "--num-tags" :: value :: tail =>
	  nextOption(map ++ Map('numTags -> value.toInt) ++ Map('wordParams -> value.toInt), tail)
	case "--transition-prior" :: value :: tail =>
	  nextOption(map ++ Map('transitionPrior -> value.toDouble), tail)
	case "--symmetric-transition-prior" :: tail =>
	  nextOption(map ++ Map('optimizeTransitionPrior -> true), tail)
	case "--emission-prior" :: value :: tail =>
	  nextOption(map ++ Map('emissionPrior -> value.toDouble), tail)
	case "--symmetric-emission-prior" :: tail =>
	  nextOption(map ++ Map('optimizeEmissionPrior -> true), tail)
	//case "--optimize-every" :: value :: tail =>
	//  nextOption(map ++ Map('optimizeEvery -> value.toInt), tail)
	//case "--annealing" :: value :: tail =>
	//  nextOption(map ++ Map('annealing -> value.toDouble), tail)
	
	// options for adaptor grammar morphology
	case "--prefix-prior" :: value :: tail =>
	  nextOption(map ++ Map('prefixPrior -> value.toDouble), tail)
	case "--tag-prior" :: value :: tail =>
	  nextOption(map ++ Map('tagPrior -> value.toDouble), tail)
	case "--base-prior" :: value :: tail =>
	  nextOption(map ++ Map('basePrior -> value.toDouble), tail)
	case "--word-prior" :: value :: tail =>
	  nextOption(map ++ Map('wordPrior -> value.toDouble), tail)
	case "--suffix-prior" :: value :: tail =>
	  nextOption(map ++ Map('suffixPrior -> value.toDouble), tail)
	case "--submorph-prior" :: value :: tail =>
	  nextOption(map ++ Map('submorphPrior -> value.toDouble), tail)
	case "--adaptor-prior-a" :: value :: tail =>
	  nextOption(map ++ Map('adaptorPriorA -> value.toDouble), tail)
	case "--adaptor-prior-b" :: value :: tail =>
	  nextOption(map ++ Map('adaptorPriorB -> value.toDouble), tail)
	case "--rule-prior" :: value :: tail =>
	  nextOption(map ++ Map('rulePrior -> value.toDouble), tail)
	case "--cache-probability" :: value :: tail =>
	  nextOption(map ++ Map('cacheProbability -> value.toInt), tail)
	//case "--word-params-prior" :: value :: tail =>
	//  nextOption(map ++ Map('wordParams -> value.toDouble), tail)


	case "--multiple-stems" :: tail =>
	  nextOption(map ++ Map('multipleStems -> true), tail)
	case "--prefixes" :: tail =>
	  nextOption(map ++ Map('prefixes -> true), tail)
	case "--suffixes" :: tail =>
	  nextOption(map ++ Map('suffixes -> true), tail)
	case "--submorphs" :: tail =>
	  nextOption(map ++ Map('submorphs -> true), tail)
	case "--non-parametric" :: tail =>
	  nextOption(map ++ Map('nonParametric -> true), tail)
	case "--hierarchical" :: tail =>
	  nextOption(map ++ Map('hierarchical -> true), tail)
	case "--use-heuristics" :: tail =>
	  nextOption(map ++ Map('useHeuristics -> true), tail)
	case "--derivational" :: tail =>
	  nextOption(map ++ Map('derivational -> true), tail)
	case "--infer-pyp" :: tail =>
	  nextOption(map ++ Map('inferPYP -> true), tail)
	case "--batch" :: tail =>
	  nextOption(map ++ Map('batch -> true), tail)



	// options for evaluation methods
	// case "--perplexity" :: value :: tail =>
	//   nextOption(map ++ Map('perplexity -> value.toInt), tail)
	// case "--variation-of-information" :: value :: tail =>
	//   nextOption(map ++ Map('variationOfInformation -> value.toInt), tail)
	// case "--best-match" :: value :: tail =>
	//   nextOption(map ++ Map('bestMatch -> value.toInt), tail)

	// other options
	case "--help" :: tail =>
	  println(usage)
	  sys.exit(0)
	//case "--test" :: tail =>
	//  runTests
	//  sys.exit(0)
        case option :: tail => 
	  println("Unknown option " + option)
	  println(usage)
	  sys.exit(1) 
      }      
    }

    // set default argument values
    val options = nextOption(Map('mode -> "joint",
				 'numBurnins -> 1,
				 'numSamples -> 10,
				 'saveEvery -> 1,
				 'markov -> 2,
				 'numTags -> 10,
				 'transitionPrior -> .1,
				 'symmetricTransitionPrior -> false,
				 'emissionPrior -> .1,
				 'symmetricEmissionPrior -> false,
				 'prefixPrior -> 1.0,
				 'wordPrior -> 1.0,
				 'suffixPrior -> 1.0,
				 'submorphPrior -> 1.0,
				 'adaptorPriorA -> 0.0,
				 'adaptorPriorB -> 100.0,
				 'rulePrior -> 1.0,
				 //'wordParams -> 1,
				 'multipleStems -> true,
				 'prefixes -> true,
				 'suffixes -> true,
				 'submorphs -> true,
				 'nonParametric -> false,
				 'typeBased -> true,
				 'basePrior -> 1.0,
				 //'w -> 1.0,
				 //'m -> 1.0,
				 //'gammaPriorA -> 1.0,
				 //'gammaPriorB -> 1.0,
				 //'betaPriorA -> 1.0,
				 //'betaPriorB -> 1.0,
				 'hierarchical -> false,
				 //'burnIn -> 10,
				 'useHeuristics -> false,
				 'derivational -> false,
				 //'batch -> false,
				 'lexGen -> 0,
				 'inferPYP -> false,
				 'tagPrior -> .1,
				 'cacheProbability -> 100,
				 'tagPrior -> .1
			       ), arglist)

    // set up logging infrastructure
    val consoleLevel = Level.INFO
    val fileLevel = Level.FINEST
    val logger = Logger.getLogger("bhmm")
    logger.setUseParentHandlers(false)
    options.get('logFile) match{
      case Some(f) => 
	val handler = new FileHandler(f.asInstanceOf[String])
	handler.setFormatter(new MyFormatter)
	handler.setLevel(fileLevel)
	logger.addHandler(handler)
      case _ => Unit
    }
    val handler = new ConsoleHandler()
    handler.setFormatter(new MyFormatter)
    handler.setLevel(consoleLevel)
    logger.addHandler(handler)
    logger.setLevel(Level.FINEST)

    // read required arguments
    val input = options.get('input).get.asInstanceOf[String]
    val output = options.get('output).get.asInstanceOf[String]    
    val numBurnins = options.get('numBurnins).get.asInstanceOf[Int]
    val numSamples = options.get('numSamples).get.asInstanceOf[Int]

    // load data
    val dataSet = DataSet.fromFile(input)
    logger.finest("%s".format(dataSet.print()))

    // construct appropriate model
    val model = (options.get('mode).get.asInstanceOf[String], options.get('typeBased).get.asInstanceOf[Boolean]) match{
      case ("tagging", true) => new TypeBasedTaggingModel(dataSet,
							  options.get('markov).get.asInstanceOf[Int],
							  options.get('numTags).get.asInstanceOf[Int],
							  options.get('transitionPrior).get.asInstanceOf[Double],
							  options.get('emissionPrior).get.asInstanceOf[Double],
							  options.get('symmetricTransitionPrior).get.asInstanceOf[Boolean],
							  options.get('symmetricEmissionPrior).get.asInstanceOf[Boolean]
							)
      case ("tagging", false) => new TokenBasedTaggingModel(dataSet,
							    options.get('markov).get.asInstanceOf[Int],
							    options.get('numTags).get.asInstanceOf[Int],
							    options.get('transitionPrior).get.asInstanceOf[Double],
							    options.get('emissionPrior).get.asInstanceOf[Double],
							    options.get('symmetricTransitionPrior).get.asInstanceOf[Boolean],
							    options.get('symmetricEmissionPrior).get.asInstanceOf[Boolean]
							  )
      case ("morphology", true) => new TypeBasedMorphologyModel(dataSet,
								options.get('numTags).get.asInstanceOf[Int],
								options.get('prefixPrior).get.asInstanceOf[Double],
								options.get('wordPrior).get.asInstanceOf[Double],
								options.get('suffixPrior).get.asInstanceOf[Double],
								options.get('submorphPrior).get.asInstanceOf[Double],
								new PitmanYorPrior(options.get('adaptorPriorA).get.asInstanceOf[Double], options.get('adaptorPriorB).get.asInstanceOf[Double]),
								options.get('basePrior).get.asInstanceOf[Double],
								options.get('tagPrior).get.asInstanceOf[Double],
								options.get('rulePrior).get.asInstanceOf[Double],
								options.get('prefixes).get.asInstanceOf[Boolean],
								options.get('suffixes).get.asInstanceOf[Boolean],
								options.get('multipleStems).get.asInstanceOf[Boolean],
								options.get('submorphs).get.asInstanceOf[Boolean],
								options.get('nonParametric).get.asInstanceOf[Boolean],
								options.get('hierarchical).get.asInstanceOf[Boolean],
								options.get('cacheProbability).get.asInstanceOf[Int],
								options.get('useHeuristics).get.asInstanceOf[Boolean],
								options.get('derivational).get.asInstanceOf[Boolean],
								options.get('inferPYP).get.asInstanceOf[Boolean]
							      )
      case ("morphology", false) => new TokenBasedMorphologyModel(dataSet,
								  options.get('numTags).get.asInstanceOf[Int],
								  options.get('prefixPrior).get.asInstanceOf[Double],
								  options.get('wordPrior).get.asInstanceOf[Double],
								  options.get('suffixPrior).get.asInstanceOf[Double],
								  options.get('submorphPrior).get.asInstanceOf[Double],
								  new PitmanYorPrior(options.get('adaptorPriorA).get.asInstanceOf[Double], options.get('adaptorPriorB).get.asInstanceOf[Double]),
								  options.get('basePrior).get.asInstanceOf[Double],
								  options.get('tagPrior).get.asInstanceOf[Double],
								  options.get('rulePrior).get.asInstanceOf[Double],
								  options.get('prefixes).get.asInstanceOf[Boolean],
								  options.get('suffixes).get.asInstanceOf[Boolean],
								  options.get('multipleStems).get.asInstanceOf[Boolean],
								  options.get('submorphs).get.asInstanceOf[Boolean],
								  options.get('nonParametric).get.asInstanceOf[Boolean],
								  options.get('hierarchical).get.asInstanceOf[Boolean],
								  options.get('cacheProbability).get.asInstanceOf[Int],
								  options.get('useHeuristics).get.asInstanceOf[Boolean],
								  options.get('derivational).get.asInstanceOf[Boolean],
								  options.get('inferPYP).get.asInstanceOf[Boolean]
								)
      case (_, true) => new TypeBasedJointModel(dataSet,
      						options.get('markov).get.asInstanceOf[Int],
      						options.get('numTags).get.asInstanceOf[Int],
      						options.get('transitionPrior).get.asInstanceOf[Double],
      						options.get('emissionPrior).get.asInstanceOf[Double],
      						options.get('symmetricTransitionPrior).get.asInstanceOf[Boolean],
      						options.get('symmetricEmissionPrior).get.asInstanceOf[Boolean],
						options.get('prefixPrior).get.asInstanceOf[Double],
						options.get('wordPrior).get.asInstanceOf[Double],
						options.get('suffixPrior).get.asInstanceOf[Double],
						options.get('submorphPrior).get.asInstanceOf[Double],
						new PitmanYorPrior(options.get('adaptorPriorA).get.asInstanceOf[Double], options.get('adaptorPriorB).get.asInstanceOf[Double]),
						options.get('basePrior).get.asInstanceOf[Double],
						options.get('tagPrior).get.asInstanceOf[Double],
						options.get('rulePrior).get.asInstanceOf[Double],
						options.get('prefixes).get.asInstanceOf[Boolean],
						options.get('suffixes).get.asInstanceOf[Boolean],
						options.get('multipleStems).get.asInstanceOf[Boolean],
						options.get('submorphs).get.asInstanceOf[Boolean],
						options.get('nonParametric).get.asInstanceOf[Boolean],
						options.get('hierarchical).get.asInstanceOf[Boolean],
						options.get('cacheProbability).get.asInstanceOf[Int],
						options.get('useHeuristics).get.asInstanceOf[Boolean],
						options.get('derivational).get.asInstanceOf[Boolean],
						options.get('inferPYP).get.asInstanceOf[Boolean]
      					      )
      case (_, false) => new TokenBasedJointModel(dataSet,
      						  options.get('markov).get.asInstanceOf[Int],
      						  options.get('numTags).get.asInstanceOf[Int],
      						  options.get('transitionPrior).get.asInstanceOf[Double],
      						  options.get('emissionPrior).get.asInstanceOf[Double],
      						  options.get('symmetricTransitionPrior).get.asInstanceOf[Boolean],
      						  options.get('symmetricEmissionPrior).get.asInstanceOf[Boolean],
						  options.get('prefixPrior).get.asInstanceOf[Double],
						  options.get('wordPrior).get.asInstanceOf[Double],
						  options.get('suffixPrior).get.asInstanceOf[Double],
						  options.get('submorphPrior).get.asInstanceOf[Double],
						  new PitmanYorPrior(options.get('adaptorPriorA).get.asInstanceOf[Double], options.get('adaptorPriorB).get.asInstanceOf[Double]),
						  options.get('basePrior).get.asInstanceOf[Double],
						  options.get('tagPrior).get.asInstanceOf[Double],
						  options.get('rulePrior).get.asInstanceOf[Double],
						  options.get('prefixes).get.asInstanceOf[Boolean],
						  options.get('suffixes).get.asInstanceOf[Boolean],
						  options.get('multipleStems).get.asInstanceOf[Boolean],
						  options.get('submorphs).get.asInstanceOf[Boolean],
						  options.get('nonParametric).get.asInstanceOf[Boolean],
						  options.get('hierarchical).get.asInstanceOf[Boolean],
						  options.get('cacheProbability).get.asInstanceOf[Int],
						  options.get('useHeuristics).get.asInstanceOf[Boolean],
						  options.get('derivational).get.asInstanceOf[Boolean],
						  options.get('inferPYP).get.asInstanceOf[Boolean]
      						)
    }

    //model.batchInitialize()
    //logger.info("%s".format(model))
    logger.info("starting burn-in")
    for(i <- 1 to numBurnins){
      logger.info("burnin #%d".format(i))
      model.sample()
      //logger.finest("%s".format(model.complete(dataSet).print()))
      //logger.info("%s".format(model))
    }
    logger.info("starting sampling")
    val out = new OutputStreamWriter(if(output.endsWith("gz")){ new GZIPOutputStream(new BufferedOutputStream(new FileOutputStream(output))) }
				     else{ new BufferedOutputStream(new FileOutputStream(output)) })
    out.write("<xml>")
    for(i <- 1 to numSamples){
      val save = (i % (options.get('saveEvery).get.asInstanceOf[Int])) == 0
      logger.info("sample #%d".format(i))
      model.sample()
      //logger.finest("%s".format(model.complete(dataSet).print()))
      //logger.info("%s".format(model))      
      if(save == true){ 
	//logger.info("saving model to %s".format(output))
	model.save(out) 
      }
    }
    out.write("</xml>")
    out.flush()
    out.close()
  }
}
