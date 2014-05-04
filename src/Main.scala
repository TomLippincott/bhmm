package bhmm

import java.util.HashMap
import java.util.Vector
import java.util.Date
import java.util.logging._
import bhmm.Test._
import bhmm.data.DataSet
import bhmm.models.{TokenBasedTaggingModel, TypeBasedTaggingModel, TokenBasedMorphologyModel, TypeBasedMorphologyModel, TokenBasedJointModel, TypeBasedJointModel}
import bhmm.evaluation.Intrinsic
import AdaptorGrammar.PitmanYorPrior
import Distributions._

object Main{  

  class MyFormatter extends Formatter{
    def format(r : LogRecord) : String = {
      val d = new Date(r.getMillis)
      d.toString + ": " + r.getMessage + "\n"
    }
  }

  val handler = new ConsoleHandler()
  handler.setFormatter(new MyFormatter)
  val logger = Logger.getLogger("bhmm")
  logger.setUseParentHandlers(false)
  logger.addHandler(handler)
  
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
    val level = Level.FINE
    handler.setLevel(level)
    logger.setLevel(level)
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

	// options for unsupervised tagger
	case "--markov" :: value :: tail =>
          nextOption(map ++ Map('markov -> value.toInt), tail)	  
	case "--num-sentences" :: value :: tail =>
          nextOption(map ++ Map('numSentences -> value.toInt), tail)	  
	case "--num-tags" :: value :: tail =>
	  nextOption(map ++ Map('numTags -> value.toInt), tail)
	case "--num-burnins" :: value :: tail =>
	  nextOption(map ++ Map('numBurnins -> value.toInt), tail)
	case "--num-samples" :: value :: tail =>
	  nextOption(map ++ Map('numSamples -> value.toInt), tail)
	case "--transition-prior" :: value :: tail =>
	  nextOption(map ++ Map('transitionPrior -> value.toDouble), tail)
	case "--symmetric-transition-prior" :: tail =>
	  nextOption(map ++ Map('optimizeTransitionPrior -> true), tail)
	case "--emission-prior" :: value :: tail =>
	  nextOption(map ++ Map('emissionPrior -> value.toDouble), tail)
	case "--symmetric-emission-prior" :: tail =>
	  nextOption(map ++ Map('optimizeEmissionPrior -> true), tail)
	case "--optimize-every" :: value :: tail =>
	  nextOption(map ++ Map('optimizeEvery -> value.toInt), tail)
	case "--annealing" :: value :: tail =>
	  nextOption(map ++ Map('annealing -> value.toDouble), tail)
	
	// options for adaptor grammar morphology
	case "--prefix-prior" :: value :: tail =>
	  nextOption(map ++ Map('prefixPrior -> value.toDouble), tail)
	case "--word-prior" :: value :: tail =>
	  nextOption(map ++ Map('wordPrior -> value.toDouble), tail)
	case "--suffix-prior" :: value :: tail =>
	  nextOption(map ++ Map('suffixPrior -> value.toDouble), tail)
	case "--sub-morph-prior" :: value :: tail =>
	  nextOption(map ++ Map('subMorphPrior -> value.toDouble), tail)
	case "--adapt-prior" :: value :: tail =>
	  nextOption(map ++ Map('adaptPrior -> value.toDouble), tail)
	case "--rule-dirichlet-prior" :: value :: tail =>
	  nextOption(map ++ Map('ruleDirichletPrior -> value.toDouble), tail)
	case "--word-params-prior" :: value :: tail =>
	  nextOption(map ++ Map('wordParams -> value.toDouble), tail)
	case "--multipleStems" :: tail =>
	  nextOption(map ++ Map('multipleStems -> true), tail)
	case "--prefixes" :: tail =>
	  nextOption(map ++ Map('prefixes -> true), tail)
	case "--suffixes" :: tail =>
	  nextOption(map ++ Map('suffixes -> true), tail)
	case "--subMorphs" :: tail =>
	  nextOption(map ++ Map('subMorphs -> true), tail)
	case "--nonParametric" :: tail =>
	  nextOption(map ++ Map('nonParametric -> true), tail)

	// options for evaluation methods
	case "--perplexity" :: value :: tail =>
	  nextOption(map ++ Map('perplexity -> value.toInt), tail)
	case "--variation-of-information" :: value :: tail =>
	  nextOption(map ++ Map('variationOfInformation -> value.toInt), tail)
	case "--best-match" :: value :: tail =>
	  nextOption(map ++ Map('bestMatch -> value.toInt), tail)

	// other options
	case "--help" :: tail =>
	  println(usage)
	  sys.exit(0)
	case "--test" :: tail =>
	  runTests
	  sys.exit(0)
        case option :: tail => 
	  println("Unknown option " + option)
	  println(usage)
	  sys.exit(1) 
      }      
    }
    
    val options = nextOption(Map('logFile -> "",
				 'mode -> "joint",
				 'markov -> 2,
				 'numSentences -> -1,
				 'numTags -> 20,
				 'numBurnins -> 10,
				 'numSamples -> 10,
				 'transitionPrior -> .1,
				 'symmetricTransitionPrior -> false,
				 'emissionPrior -> .1,
				 'symmetricEmissionPrior -> false,
				 'optimizeEvery -> 5,
				 'annealing -> 1.0,
				 'perplexity -> 0,
				 'bestMatch -> 0,
				 'variationOfInformation -> 0,
				 'prefixPrior -> .001,
				 'wordPrior -> .001,
				 'suffixPrior -> .001,
				 'subMorphPrior -> .001,
				 'adaptorPriorA -> 0.0,
				 'adaptorPriorB -> 1.0,
				 'ruleDirichletPrior -> .1,
				 'wordParams -> 1,
				 'multipleStems -> true,
				 'prefixes -> true,
				 'suffixes -> true,
				 'subMorphs -> true,
				 'nonParametric -> true,
				 'typeBased -> true,
				 'basePrior -> 1.0,
				 'gammaPriorA -> 1.0,
				 'gammaPriorB -> 1.0,
				 'betaPriorA -> 1.0,
				 'betaPriorB -> 1.0,
				 'isPrefixAllowed -> false,
				 'isSuffixAllowed -> false,
				 'isSubMorphAllowed -> false,
				 'isMultipleStemsAllowed -> false,
				 'isNonParametric -> false,
				 'isHierarchical -> false,
				 'burnIn -> 0,
				 'useHeuristics -> false,
				 'isDerivational -> false,
				 'isBatch -> false,
				 'lexGen -> 1000,
				 'inferePYP -> false,
				 'w -> 1.0,
				 'm -> 1.0,
				 'tagPrior -> 1.0,
				 'cachingProb -> 10
			       ), arglist)


    val output = options.get('output).get.asInstanceOf[String]
    //val printPerplexity = options.get('perplexity).get.asInstanceOf[Int]
    //val printVariationOfInformation = options.get('variationOfInformation).get.asInstanceOf[Int]
    //val printBestMatch = options.get('bestMatch).get.asInstanceOf[Int]
    val numBurnins = options.get('numBurnins).get.asInstanceOf[Int]
    val numSamples = options.get('numSamples).get.asInstanceOf[Int]
    val optimizeEvery = options.get('optimizeEvery).get.asInstanceOf[Int]
    val annealing = options.get('annealing).get.asInstanceOf[Double]

    val ruleDirichletPrior = options.get('ruleDirichletPrior).get.asInstanceOf[Double]
    val wordParams = options.get('wordParams).get.asInstanceOf[Int]
    val prefixes = options.get('prefixes).get.asInstanceOf[Boolean]
    val suffixes = options.get('suffixes).get.asInstanceOf[Boolean]
    val multipleStems = options.get('multipleStems).get.asInstanceOf[Boolean]
    val subMorphs = options.get('subMorphs).get.asInstanceOf[Boolean]
    val nonParametric = options.get('nonParametric).get.asInstanceOf[Boolean]

    val dataSet = DataSet.fromFile(options.get('input).get.asInstanceOf[String],
				   options.get('numSentences).get.asInstanceOf[Int],
				   options.get('markov).get.asInstanceOf[Int])

    logger.fine("%s".format(dataSet))

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
								options.get('prefixPrior).get.asInstanceOf[Double],
								options.get('wordPrior).get.asInstanceOf[Double],
								options.get('suffixPrior).get.asInstanceOf[Double],
								options.get('subMorphPrior).get.asInstanceOf[Double],
								new PitmanYorPrior(options.get('adaptorPriorA).get.asInstanceOf[Double], options.get('adaptorPriorB).get.asInstanceOf[Double]),
								options.get('basePrior).get.asInstanceOf[Double],
								options.get('tagPrior).get.asInstanceOf[Double],
								options.get('ruleDirichletPrior).get.asInstanceOf[Double],
								options.get('wordParams).get.asInstanceOf[Int],
								options.get('isPrefixAllowed).get.asInstanceOf[Boolean],
								options.get('isSuffixAllowed).get.asInstanceOf[Boolean],
								options.get('isMultipleStemsAllowed).get.asInstanceOf[Boolean],
								options.get('isSubMorphAllowed).get.asInstanceOf[Boolean],
								options.get('isNonParametric).get.asInstanceOf[Boolean],
								options.get('isHierarchical).get.asInstanceOf[Boolean],
								options.get('burnIn).get.asInstanceOf[Int],
								options.get('cachingProb).get.asInstanceOf[Int],
								options.get('useHeuristics).get.asInstanceOf[Boolean],
								options.get('lexGen).get.asInstanceOf[Int],
								options.get('isDerivational).get.asInstanceOf[Boolean],
								options.get('inferePYP).get.asInstanceOf[Boolean],
								options.get('w).get.asInstanceOf[Double],
								options.get('m).get.asInstanceOf[Double],
								new GammaDistribution(options.get('gammaPriorA).get.asInstanceOf[Double], options.get('gammaPriorB).get.asInstanceOf[Double]),
								new BetaDistribution(options.get('betaPriorA).get.asInstanceOf[Double], options.get('betaPriorB).get.asInstanceOf[Double]),
								options.get('isBatch).get.asInstanceOf[Boolean]
							      )

      case ("morphology", false) => new TokenBasedMorphologyModel(dataSet,
								  options.get('prefixPrior).get.asInstanceOf[Double],
								  options.get('wordPrior).get.asInstanceOf[Double],
								  options.get('suffixPrior).get.asInstanceOf[Double],
								  options.get('subMorphPrior).get.asInstanceOf[Double],
								  new PitmanYorPrior(options.get('adaptorPriorA).get.asInstanceOf[Double], options.get('adaptorPriorB).get.asInstanceOf[Double]),
								  options.get('basePrior).get.asInstanceOf[Double],
								  options.get('tagPrior).get.asInstanceOf[Double],
								  options.get('ruleDirichletPrior).get.asInstanceOf[Double],
								  options.get('wordParams).get.asInstanceOf[Int],
								  options.get('isPrefixAllowed).get.asInstanceOf[Boolean],
								  options.get('isSuffixAllowed).get.asInstanceOf[Boolean],
								  options.get('isMultipleStemsAllowed).get.asInstanceOf[Boolean],
								  options.get('isSubMorphAllowed).get.asInstanceOf[Boolean],
								  options.get('isNonParametric).get.asInstanceOf[Boolean],
								  options.get('isHierarchical).get.asInstanceOf[Boolean],
								  options.get('burnIn).get.asInstanceOf[Int],
								  options.get('cachingProb).get.asInstanceOf[Int],
								  options.get('useHeuristics).get.asInstanceOf[Boolean],
								  options.get('lexGen).get.asInstanceOf[Int],
								  options.get('isDerivational).get.asInstanceOf[Boolean],
								  options.get('inferePYP).get.asInstanceOf[Boolean],
								  options.get('w).get.asInstanceOf[Double],
								  options.get('m).get.asInstanceOf[Double],
								  new GammaDistribution(options.get('gammaPriorA).get.asInstanceOf[Double], options.get('gammaPriorB).get.asInstanceOf[Double]),
								  new BetaDistribution(options.get('betaPriorA).get.asInstanceOf[Double], options.get('betaPriorB).get.asInstanceOf[Double]),
								  options.get('isBatch).get.asInstanceOf[Boolean]
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
						options.get('subMorphPrior).get.asInstanceOf[Double],
						new PitmanYorPrior(options.get('adaptorPrior).get.asInstanceOf[Double], options.get('adaptorPrior).get.asInstanceOf[Double]),
						options.get('basePrior).get.asInstanceOf[Double],
						options.get('tagPrior).get.asInstanceOf[Double],
						options.get('ruleDirichletPrior).get.asInstanceOf[Double],
						options.get('wordParams).get.asInstanceOf[Int],
						options.get('isPrefixAllowed).get.asInstanceOf[Boolean],
						options.get('isSuffixAllowed).get.asInstanceOf[Boolean],
						options.get('isMultipleStemsAllowed).get.asInstanceOf[Boolean],
						options.get('isSubMorphAllowed).get.asInstanceOf[Boolean],
						options.get('isNonParametric).get.asInstanceOf[Boolean],
						options.get('isHierarchical).get.asInstanceOf[Boolean],
						options.get('burnIn).get.asInstanceOf[Int],
						options.get('cachingProb).get.asInstanceOf[Int],
						options.get('useHeuristics).get.asInstanceOf[Boolean],
						options.get('lexGen).get.asInstanceOf[Int],
						options.get('isDerivational).get.asInstanceOf[Boolean],
						options.get('inferePYP).get.asInstanceOf[Boolean],
						options.get('w).get.asInstanceOf[Double],
						options.get('m).get.asInstanceOf[Double],
						new GammaDistribution(options.get('gamma).get.asInstanceOf[Double], options.get('gamma).get.asInstanceOf[Double]),
						new BetaDistribution(options.get('beta).get.asInstanceOf[Double], options.get('beta).get.asInstanceOf[Double]),
						options.get('isBatch).get.asInstanceOf[Boolean]
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
						  options.get('subMorphPrior).get.asInstanceOf[Double],
						  new PitmanYorPrior(options.get('adaptorPrior).get.asInstanceOf[Double], options.get('adaptorPrior).get.asInstanceOf[Double]),
						  options.get('basePrior).get.asInstanceOf[Double],
						  options.get('tagPrior).get.asInstanceOf[Double],
						  options.get('ruleDirichletPrior).get.asInstanceOf[Double],
						  options.get('wordParams).get.asInstanceOf[Int],
						  options.get('isPrefixAllowed).get.asInstanceOf[Boolean],
						  options.get('isSuffixAllowed).get.asInstanceOf[Boolean],
						  options.get('isMultipleStemsAllowed).get.asInstanceOf[Boolean],
						  options.get('isSubMorphAllowed).get.asInstanceOf[Boolean],
						  options.get('isNonParametric).get.asInstanceOf[Boolean],
						  options.get('isHierarchical).get.asInstanceOf[Boolean],
						  options.get('burnIn).get.asInstanceOf[Int],
						  options.get('cachingProb).get.asInstanceOf[Int],
						  options.get('useHeuristics).get.asInstanceOf[Boolean],
						  options.get('lexGen).get.asInstanceOf[Int],
						  options.get('isDerivational).get.asInstanceOf[Boolean],
						  options.get('inferePYP).get.asInstanceOf[Boolean],
						  options.get('w).get.asInstanceOf[Double],
						  options.get('m).get.asInstanceOf[Double],
						  new GammaDistribution(options.get('gamma).get.asInstanceOf[Double], options.get('gamma).get.asInstanceOf[Double]),
						  new BetaDistribution(options.get('beta).get.asInstanceOf[Double], options.get('beta).get.asInstanceOf[Double]),
						  options.get('isBatch).get.asInstanceOf[Boolean]
						)
    }

    //model.batchInitialize()
    
    for(i <- 1 to numBurnins){
      logger.fine("burnin #%d".format(i))
      model.sample()
      logger.fine("Joint latent probability: %s".format(model.totalProbability()))
      logger.fine("%s".format(model))
    }
    for(i <- 1 to numSamples){
      logger.fine("sample #%d".format(i))
      model.sample()
      logger.fine("Joint latent probability: %s".format(model.totalProbability()))
      logger.fine("%s".format(model))
    }
    model.save(output)

  }
}
