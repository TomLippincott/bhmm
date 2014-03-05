package bhmm

import java.util.HashMap
import java.util.Vector
import java.util.Date
import java.util.logging._
import bhmm.Probability._
import bhmm.Dirichlet._
import bhmm.Types._
import bhmm.Utilities._
import bhmm.Test._
import bhmm.DataSet._
import bhmm.Metrics._
import bhmm.Morphology._

object Main{  

  class MyFormatter extends Formatter{
    def format(r : LogRecord) : String = {
      val d = new Date(r.getMillis)
      d.toString + ": " + r.getMessage + "\n"
    }
  }

  val handler = new ConsoleHandler()
  handler.setFormatter(new MyFormatter)
  val logger = Logger.getAnonymousLogger()
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
        case "--input" :: value :: tail =>
          nextOption(map ++ Map('input -> value), tail)
        case "--output" :: value :: tail =>
          nextOption(map ++ Map('output -> value), tail)

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
	case "--optimize-transition-prior" :: value :: tail =>
	  nextOption(map ++ Map('optimizeTransitionPrior -> value), tail)
	case "--emission-prior" :: value :: tail =>
	  nextOption(map ++ Map('emissionPrior -> value.toDouble), tail)
	case "--optimize-emission-prior" :: value :: tail =>
	  nextOption(map ++ Map('optimizeEmissionPrior -> value), tail)

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
    
    val options = nextOption(Map('markov -> 2,
				 'numSentences -> 100,
				 'numTags -> 20,
				 'numBurnins -> 10,
				 'numSamples -> 10,
				 'transitionPrior -> .1,
				 'optimizeTransitionPrior -> "no",
				 'emissionPrior -> .1,
				 'optimizeEmissionPrior -> "no",
				 'perplexity -> 0,
				 'bestMatch -> 0,
				 'variationOfInformation -> 0,
				 'prefixPrior -> .001,
				 'wordPrior -> .001,
				 'suffixPrior -> .001,
				 'subMorphPrior -> .001,
				 'adaptPrior -> 250.0,
				 'ruleDirichletPrior -> .1,
				 'wordParams -> 1,
				 'multipleStems -> true,
				 'prefixes -> true,
				 'suffixes -> true,
				 'subMorphs -> true,
				 'nonParametric -> true), arglist)

    val printPerplexity = options.get('perplexity).get.asInstanceOf[Int]
    val printVariationOfInformation = options.get('variationOfInformation).get.asInstanceOf[Int]
    val printBestMatch = options.get('bestMatch).get.asInstanceOf[Int]
    val input = options.get('input).get.asInstanceOf[String]
    val output = options.get('output).get.asInstanceOf[String]
    val markov = options.get('markov).get.asInstanceOf[Int]
    val numSentences = options.get('numSentences).get.asInstanceOf[Int]
    val numTags = options.get('numTags).get.asInstanceOf[Int]
    val numBurnins = options.get('numBurnins).get.asInstanceOf[Int]
    val numSamples = options.get('numSamples).get.asInstanceOf[Int]
    val optimizeEmissionPrior = options.get('optimizeEmissionPrior).get.asInstanceOf[String]
    val optimizeTransitionPrior = options.get('optimizeTransitionPrior).get.asInstanceOf[String]

    
    val ruleDirichletPrior = options.get('ruleDirichletPrior).get.asInstanceOf[Double]
    val wordParams = options.get('wordParams).get.asInstanceOf[Int]
    val prefixes = options.get('prefixes).get.asInstanceOf[Boolean]
    val suffixes = options.get('suffixes).get.asInstanceOf[Boolean]
    val multipleStems = options.get('multipleStems).get.asInstanceOf[Boolean]
    val subMorphs = options.get('subMorphs).get.asInstanceOf[Boolean]
    val nonParametric = options.get('nonParametric).get.asInstanceOf[Boolean]

    val wordToIndex = scala.collection.mutable.Map[String, Int]()
    val tagToIndex = scala.collection.mutable.Map[String, Int]()

    val dataSet = parseInput(numSentences, numTags, markov, input, wordToIndex, tagToIndex)
    val indexToWord = wordToIndex.map(_.swap)
    val indexToTag = tagToIndex.map(_.swap)
    val numGoldTags = tagToIndex.size
    def printWords(ixs : Seq[Int]) : Unit = { ixs.map(x => if(x != -1){print(indexToWord(x))}else{print(-1)}) }
    
    val numWords = wordToIndex.size
    val numHistories = scala.math.pow((numTags.toDouble + 1), markov.toDouble).toInt
    
    val tagByWord = Array.fill[Int](numTags, numWords)(0)
    val samples = Array.fill[Int](numSamples, numTags, numWords)(0)
    val historyByTag = Array.fill[Int](numHistories, numTags + 1)(0)
    val historyCounts = Array.fill[Int](numHistories)(0)
    val tagCounts = Array.fill[Int](numTags + 1)(0)
    val wordCounts = Array.fill[Int](numWords)(0)

    dataSet.filter(_._1 != -1).map(x => wordCounts(x._1) += 1)
    logger.fine("total words in data set: %d".format(wordCounts.sum))
    
    tagCounts(numTags) = dataSet.filter(x => x._2 == numTags).length

    var transitionPriors = (0 to numTags).map(_ => options.get('transitionPrior).get.asInstanceOf[Double]).toArray
    var emissionPriors = (0 to numWords - 1).map(_ => options.get('emissionPrior).get.asInstanceOf[Double]).toArray

    val rng = new scala.util.Random()
        
    def fullSample(data : DataSet, number : Int) : Unit = {
      def crement(index : Int, amount : Int) : Unit = {
	data(index) match{
	  case (-1, _, _) => logger.finest("no word here, so no emission adjustment")
	  case (_, -1, _) => logger.finest("no tag here, so no emission adjustment")
	  case (wordId, tagId, _) =>
	    logger.finest("adjusting tag=%d, word=%s count by %d".format(tagId, indexToWord(wordId), amount))
	    tagByWord(tagId)(wordId) += amount
	    logger.finest("adjusting tag=%d count by %d".format(tagId, amount))
	    tagCounts(tagId) += amount
	}

	val windows = (index - markov to math.min(data.length - 1, index + markov)).sliding(markov + 1).toList
	val wTags = windows.map(w => w.map(data(_)._2)).filter(x => x.count(_ == -1) == 0)
	val htPairs = wTags.map(x => (historyToIndex(x.take(markov), numTags + 1), x.last))
	htPairs.map(x => logger.finest("adjusting history=%s, tag=%s count by %d".format(x._1, x._2, amount)))
	htPairs.map(x => (historyCounts(x._1) += amount, historyByTag(x._1)(x._2) += amount))
      }

      def oneDensity(windows : Seq[Seq[Int]]) : Prob = {
	logger.finest("windows: %s".format(windows))
	val locations = windows.map(w => w.map(data(_)))
	logger.finest("locations: %s".format(locations))
	val contexts = windows.map(w => w.map(data(_)._2)).map(x => (historyToIndex(x.take(markov), numTags + 1), x.last))
	if(contexts.last._2 == -1 || contexts.last._2 == numTags){
	  fromProb(1.0 / numTags)
	}else{
	  logger.finest("contexts: %s".format(contexts))
	  val num = historyByTag(contexts.last._1)(contexts.last._2) + contexts.dropRight(1).map{x => if(x._1 == contexts.last._1 && x._2 == contexts.last._2){ 1 }else{ 0 }}.sum
	  val den = historyCounts(contexts.last._1) + contexts.dropRight(1).map{x => if(x._1 == contexts.last._1){ 1 }else{ 0 }}.sum
	  
	  fromProb((num + transitionPriors(contexts.last._2)) / (numTags + transitionPriors.sum))
	}  
      }

      def transitionDist(windows : Seq[Seq[Int]], offset : Int) : Dist = {
	val primaryWindowTags = data.slice(windows(0).head, windows(0).last + 1).map(_._2)
	val target = data(windows(0).last)
	val res = if(primaryWindowTags.count(_ == -1) > 1 || primaryWindowTags.last == numTags){ (1 to numTags).map(_ => fromProb(1.0 / numTags)) }else{
	  (0 to numTags - 1).map{ x =>
	     data(windows(0).last) = (target._1, x, target._3)
	     oneDensity(windows.take(x + 1))
			       }
	}
	//logger.fine("%s".format(res))
	res
      }

      def emissionDist(wordId : Int) : Dist = {
	val priorSum = emissionPriors.sum
	val prior = emissionPriors(wordId)
	val counts = tagByWord.map(_(wordId).toDouble)
	val countSum = counts.sum
	counts.map(x => fromProb((x + prior) / (prior * numTags + countSum)))
      }

      def oneSample(index : Int) : Unit = {
	val loc = data(index)
	logger.finest("sampling at index %d = %s".format(index, loc))
        loc match{
	  case (-1, _, _) =>
	    Unit
	  case (wordId, tagId, goldTagId) =>
	    crement(index, -1)
	    val eDist = emissionDist(wordId)
	    logger.finest("%s".format(eDist))	 
	    val fromIndex = index - markov
	    val toIndex = math.min(data.length - 1, index + markov)
	    val windows = (fromIndex to toIndex).sliding(markov + 1).toList
	    logger.finest("unfiltered windows: %s".format(windows))
	    val tDists = (0 to windows.length - 1).map(x => transitionDist(windows.slice(0, x + 1), x))
	    val fDist = tDists.fold(eDist)(times)
	    val newTag = fDist.sample(rng.nextDouble)
	    logger.finest("assigning new tag %d".format(newTag))
	    data(index) = (wordId, newTag, goldTagId)	    
	    crement(index, 1)
	}
      }

      def perplexity() : Double = {
	def wordDensity(i : Int, word : Int) : Prob = {	  
	  val history = historyToIndex(data.slice(i - markov, i).map(_._2), numTags + 1)
	  val transDist = (0 to numTags - 1).map(tag => fromProb((historyByTag(history)(tag) + transitionPriors(tag)) / (historyCounts(history) + transitionPriors.sum)) *  
					                fromProb((tagByWord(tag)(word) + emissionPriors(word)) / (tagCounts(tag) + emissionPriors.sum)))
	  val d = transDist.drop(1).fold(transDist(0))((x, y) => x + y)
	  d
	}
	// normalized
	//val probs = (0 to data.length - 1).filter(data(_)._1 != -1).map(i => new Dist((0 to numWords - 1).map(wordDensity(i, _))).ps(data(i)._1).l)
	// unnormalized
	val probs = (0 to data.length - 1).filter(data(_)._1 != -1).map(i => wordDensity(i, data(i)._1).l)
	expbase(- probs.sum / probs.length)
      }
      

      (0 to data.length - 1).map(oneSample(_))
      logger.fine("tag counts: %s".format(tagCounts.toList))
      val optimizeIterations = 200

      optimizeTransitionPrior match{
	case "symmetric" => 
	  symmetricOptimizeHyperParameters(transitionPriors, historyByTag, optimizeIterations)
	  logger.fine("optimized symmetric transition priors: %d x %f".format(transitionPriors.length, transitionPriors.sum / transitionPriors.length))
	case "asymmetric" => 
	  optimizeHyperParameters(transitionPriors, historyByTag, optimizeIterations)
	  logger.fine("optimized asymmetric transition priors: average = %f".format(transitionPriors.sum / transitionPriors.length))
	case _ =>
	  Unit
      }

      optimizeEmissionPrior match{
	case "symmetric" => 
	  symmetricOptimizeHyperParameters(emissionPriors, tagByWord, optimizeIterations)
	  logger.fine("optimized symmetric emission priors: %d x %f".format(emissionPriors.length, emissionPriors.sum / emissionPriors.length))
	case "asymmetric" => 
	  optimizeHyperParameters(emissionPriors, tagByWord, optimizeIterations)
	  logger.fine("optimized asymmetric emission priors: average = %f".format(emissionPriors.sum / emissionPriors.length))
	case _ =>
	  Unit
      }
      //if(printPerplexity > 0 && number % printPerplexity == 0) logger.fine("Perplexity: %f".format(perplexity()))
      //if(printBestMatch > 0 && number % printBestMatch == 0) logger.fine("Best-Match F-Score: %f".format(evaluateGreedyOneToOne(data)))
      //if(printVariationOfInformation > 0 && number % printVariationOfInformation == 0) logger.fine("Variation of Information: %f".format(variationOfInformation()))
    }

    for(i <- 1 until numBurnins + 1){
      logger.fine("burnin #%d".format(i))
      fullSample(dataSet, i)
    }
    for(i <- 1 until numSamples + 1){
      logger.fine("sample #%d".format(i))
      fullSample(dataSet, i + numBurnins)
      samples(i - 1) = tagByWord
    }
    val summedTagByWord = Array.fill[Int](numTags, numWords)(0)
    (0 to numTags - 1).map(t => (0 to numWords - 1).map(w => (0 to numSamples - 1).map(s => summedTagByWord(t)(w) += samples(s)(t)(w))))
    val wordTotals = (0 to numWords - 1).map(w => summedTagByWord.map(_(w)).sum)

    summedTagByWord.map(printTopic(_))

    def printTopic(row : Seq[Int], count : Int = 10) : Unit = {
      logger.info("%s".format(row.zip(0 to row.size).sorted.reverse.toList.filter(_._1 > 0).map(x => (indexToWord(x._2), x._1, wordTotals(x._2))).take(count)))
    }
  }
}
