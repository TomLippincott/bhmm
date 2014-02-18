import java.io.File
import Accessories._
import java.io._
import java.util.HashMap
import java.util.Vector
import java.util.Date
import java.util.zip._
import java.util.logging._
import CFG.NonTerminal
import CFG.Rule
import CFG.RuleRightHand
import CFG.Terminal
import Data._
import AdaptorGrammar.AdaptorSet
import AdaptorGrammar.RuleSet
import Parser.CKY
import Sampler.AdaptorSampler

object BHMM{  

  val coarseTags = Map("CONJ" -> Seq("CC"),
		       "DET" -> Seq("DT", "PDT"),
		       "INPUNC" -> Seq("$", ",", ":", "LS", "SYM", "UH"),
		       "ENDPUNC" -> Seq("."),
		       "LPUNC" -> Seq("-LRB-", "``"),
		       "POS" -> Seq("POS"),
		       "PRT" -> Seq("RP"),
		       "TO" -> Seq("TO"),
		       "PREP" -> Seq("IN"),
		       "RPUNC" -> Seq("-RRB-", "''"),
		       "W" -> Seq("WDT", "WP$", "WP", "WRB"),
		       // open-class
		       "V" -> Seq("MD", "VBD", "VBP", "VB", "VBZ"),
		       "ADJ" -> Seq("CD", "JJ", "JJR", "JJS", "PRP$"),
		       "VBG" -> Seq("VBG"),
		       "ADV" -> Seq("RB", "RBR", "RBS"),
		       "N" -> Seq("EX", "FW", "NN", "NNP", "NNPS", "NNS", "PRP"),
		       "VBN" -> Seq("VBN"))

  val fineToCoarseTag = coarseTags.map(x => x._2.map((_, x._1))).fold(Seq())((x, y) => x ++ y).toMap

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
  
  /*
   * BEGIN Probability
   */  

  val base = 10
  def logbase(x: Double) = scala.math.log(x)/scala.math.log(base)
  def expbase(x: Double) = scala.math.pow(base.toDouble, x)

  def fromProb(x : Double) : Prob = new Prob(logbase(x))
  def fromLog(x : Double) : Prob = new Prob(x)

  class Prob(x : Double) extends Ordered[Prob]{
    def compare(that : Prob) = if(this.l - that.l < 0){ -1 }else if(this.l == that.l){ 0 }else{ 1 }
    def l = x
    def *(y : Prob) : Prob = new Prob(l + y.l)
    def /(y : Prob) : Prob = new Prob(l - y.l)
    def inverse = fromProb(1.0) / this
    def +(y : Prob) : Prob = {
      def go(a : Double, b : Double) : Prob = {
	val d = a - b
	if(d < -20){ new Prob(l) }else{ new Prob(b + logbase(1.0 + expbase(d))) }
      }
      if(l < y.l){
	go(l, y.l) 
      }
      else{ 
	go(y.l, l) 
      }
    }
    override def toString = "Prob(%f/%f)".format(expbase(l), l)
  }

  class Dist(vs : Seq[Prob]){
    val scale = new Prob(vs.tail.fold(vs.head)((x, y) => x + y).l).inverse
    def ps = vs.map(x => x * scale)
    def sample(d : Double) : Int = {
      val p = fromProb(d)
      val ms = ps.tail.scanLeft(ps.head)((x, y) => x + y)
      ms.lastIndexWhere((x => p > x)) + 1
    }
    def *(ys : Dist) : Dist = {
      ps.zip(ys.ps).map(x => x._1 * x._2)
    }
    def +(ys : Dist) : Dist = {
      ps.zip(ys.ps).map(x => x._1 + x._2)
    }
    override def toString = "Dist(%s)".format(ps)
  }
  
  def times(a : Dist, b : Dist) : Dist = {
    a * b
  }

  def sum(a : Dist, b : Dist) : Dist = {
    a + b
  }

  implicit def ProbSeqToDist(xs : Seq[Prob]) : Dist = new Dist(xs)
  implicit def ProbArrayToDist(xs : Array[Prob]) : Dist = new Dist(xs)

  /*
   * END Probability
   */

  val usage = """
  All arguments are optional:
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

  type Location = Tuple3[Int, Int, Int]
  type Lookup = scala.collection.mutable.Map[String, Int]
  type VectorCounts = scala.collection.mutable.ArraySeq[Double]
  type Counts = scala.collection.mutable.ArraySeq[VectorCounts]  
  type DataSet = scala.collection.mutable.ArraySeq[Location]


  def parseInput(numLines: Int, numTags : Int, markov: Int, filename: String, wordLookup: Lookup, tagLookup: Lookup) : DataSet = {
    val nullLoc = (-1, numTags, 0)
    def parseLocation(loc: String): Location = {
      val toks = loc.split("/")
      val word = toks.dropRight(1).mkString("/")
      val tag = fineToCoarseTag(toks.last)
      val wordId = if(wordLookup.contains(word)){ wordLookup(word) }else{ wordLookup += Tuple2(word, wordLookup.size); wordLookup(word) }
      val goldTagId = if(tagLookup.contains(tag)){ tagLookup(tag) }else{ tagLookup += Tuple2(tag, tagLookup.size); tagLookup(tag) }
      val tagId = -1
      (wordId, tagId, goldTagId)
    }
    def parseSentence(line: String): Seq[Location] = {
      (0 until markov).map(_ => nullLoc) ++ line.split(" ").map(parseLocation(_))
    }
    val src = io.Source.fromInputStream(new GZIPInputStream(new BufferedInputStream(new FileInputStream(filename))))
    collection.mutable.ArraySeq[Location](src.getLines().take(numLines).map(parseSentence(_)).toList.flatten.toArray : _*)
  }

  def runTests = {

    val rng = new scala.util.Random()

    val xs = (1 to 5).map(x => rng.nextDouble / 5)
    val xdist = new Dist(xs.map(fromProb(_)))

    val ys = (1 to 5).map(x => rng.nextDouble / 5)
    val ydist = new Dist(ys.map(fromProb(_)))

    println(xdist)
    val results = Array.fill[Int](5)(0)    
    (1 to 10000).map(x => rng.nextDouble).map(x => results(xdist.sample(x)) += 1)
    println(results.toList)


    println(ydist)
    val results2 = Array.fill[Int](5)(0)    
    (1 to 10000).map(x => rng.nextDouble).map(x => results2(ydist.sample(x)) += 1)
    println(results2.toList)

    println(xdist * ydist)
    println(xdist + ydist)

  }

  def morphology(config: String) = {
  }
  
  def main(args: Array[String]){
    type OptionMap = Map[Symbol, Any]
    if(args.length == 0) println(usage)
    val arglist = args.toList
    def nextOption(map : OptionMap, list: List[String]) : OptionMap = {
      def isSwitch(s : String) = (s(0) == '-')
      list match {
        case Nil => map
        case "--config" :: value :: tail =>
          nextOption(map ++ Map('config -> value), tail)
        case "--input" :: value :: tail =>
          nextOption(map ++ Map('input -> value), tail)
        case "--output" :: value :: tail =>
          nextOption(map ++ Map('output -> value), tail)
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
	case "--emission-prior" :: value :: tail =>
	  nextOption(map ++ Map('emissionPrior -> value.toDouble), tail)
	case "--perplexity" :: tail =>
	  nextOption(map ++ Map('perplexity -> true), tail)
	case "--help" :: tail =>
	  println(usage)
	  sys.exit(0)
	case "--test" :: tail =>
	  runTests
	  sys.exit(0)
        case option :: tail => 
	  println("Unknown option "+option)
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
				 'emissionPrior -> .1,
				 'perplexity -> false), arglist)

    val level = Level.FINE
    handler.setLevel(level)
    logger.setLevel(level)

    val printPerplexity = options.get('perplexity).get.asInstanceOf[Boolean]
    val config = options.get('config).get.asInstanceOf[String]
    val input = options.get('input).get.asInstanceOf[String]
    val output = options.get('input).get.asInstanceOf[String]
    val markov = options.get('markov).get.asInstanceOf[Int]
    val numSentences = options.get('numSentences).get.asInstanceOf[Int]
    val numTags = options.get('numTags).get.asInstanceOf[Int]
    val numBurnins = options.get('numBurnins).get.asInstanceOf[Int]
    val numSamples = options.get('numSamples).get.asInstanceOf[Int]
    
    val wordToIndex = scala.collection.mutable.Map[String, Int]()
    val tagToIndex = scala.collection.mutable.Map[String, Int]()

    val dataSet = parseInput(numSentences, numTags, markov, input, wordToIndex, tagToIndex)
    val indexToWord = wordToIndex.map(_.swap)
    val indexToTag = tagToIndex.map(_.swap)
    def printWords(ixs : Seq[Int]) : Unit = { ixs.map(x => if(x != -1){print(indexToWord(x))}else{print(-1)}) }
    
    val numWords = wordToIndex.size
    val numHistories = scala.math.pow((numTags.toDouble + 1), markov.toDouble).toInt
    
    val tagByWord = Array.fill[Int](numTags, numWords)(0)
    val historyByTag = Array.fill[Int](numHistories, numTags + 1)(0)
    val historyCounts = Array.fill[Int](numHistories)(0)
    val tagCounts = Array.fill[Int](numTags + 1)(0)
    val wordCounts = Array.fill[Int](numWords)(0)

    dataSet.filter(_._1 != -1).map(x => wordCounts(x._1) += 1)
    logger.fine("total words in data set: %d".format(wordCounts.sum))
    
    tagCounts(numTags) = dataSet.filter(x => x._2 == numTags).length

    var transitionPriors = (0 to numTags).map(_ => options.get('transitionPrior).get.asInstanceOf[Double]).toArray
    var emissionPriors = (0 to numWords).map(_ => options.get('emissionPrior).get.asInstanceOf[Double]).toArray

    def evaluateOneToOne(data : DataSet) : Double = {
      val cm = Array.fill[Int](numTags, tagToIndex.size)(0)
      data.filter(x => x._1 != -1).map(x => cm(x._2)(x._3) += 1)
      val learnedToGold = scala.collection.mutable.Map[Int, Int]()
      val largest = cm.map{x => 
	val numCorrect = x.max
	val goldTag = x.indexWhere(_ == numCorrect)
	val numGuesses = x.sum
	val numTargets = cm.map(_(goldTag)).sum
	(goldTag, numCorrect.toDouble, numGuesses.toDouble, numTargets.toDouble)
			 }
      val fs = largest.filter(x => x._3 > 0 && x._4 > 0).map(x => (x._2 / x._3, x._2 / x._4)).filter(x => x._1 > 0 && x._2 > 0).map(x => 2 * (x._1 * x._2) / (x._1 + x._2))
      fs.sum / fs.length
    }

    morphology(config)
    
    val rng = new scala.util.Random()
    
    def indexToHistory(i : Int, acc : Seq[Int] = Seq()) : Seq[Int] = {
      i match{
	case 0 =>
	  if(acc.length < markov){ indexToHistory(0, 0 +: acc) }else{ acc }
	case _ =>
	  val c = scala.math.pow(numTags.toDouble + 1, markov - acc.length - 1).toInt
	  val d = i / c
	  indexToHistory(i - (c * d), d +: acc)
      }
    }
    
    def historyToIndex(history: Seq[Int]) : Int = {
      logger.finest("history: %s".format(history))
      val i = history.zipWithIndex.map(x => x._1 * scala.math.pow(numTags.toDouble + 1, x._2.toDouble)).sum.toInt
      logger.finest("index: %s".format(i))
      i
    }
    
    def fullSample(data : DataSet) : Unit = {
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
	val htPairs = wTags.map(x => (historyToIndex(x.take(markov)), x.last))
	htPairs.map(x => logger.finest("adjusting history=%s, tag=%s count by %d".format(x._1, x._2, amount)))
	htPairs.map(x => (historyCounts(x._1) += amount, historyByTag(x._1)(x._2) += amount))
      }

      def oneDensity(windows : Seq[Seq[Int]]) : Prob = {
	logger.finest("windows: %s".format(windows))
	val locations = windows.map(w => w.map(data(_)))
	logger.finest("locations: %s".format(locations))
	val contexts = windows.map(w => w.map(data(_)._2)).map(x => (historyToIndex(x.take(markov)), x.last))
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
	if(primaryWindowTags.count(_ == -1) > 1 || primaryWindowTags.last == numTags){ (1 to numTags).map(_ => fromProb(1.0 / numTags)) }else{
	  (0 to numTags - 1).map{ x =>
	     data(windows(0).last) = (target._1, x, target._3)
	     oneDensity(windows.take(x + 1))
			       }
	}
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
	    data(index) = (wordId, newTag, goldTagId)	    
	    crement(index, 1)
	}
      }

      def perplexity() : Double = {
	def wordDensity(i : Int, word : Int) : Prob = {	  
	  val history = historyToIndex(data.slice(i - markov, i).map(_._2))
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

      def variationOfInformation() : Double = {
	1.0
      }

      def optimizeHyperParameters(parameters : Array[Double], counts : Array[Array[Int]], sumCounts : Array[Int]) : Unit = {
	logger.finer("pre-optimization: %s".format(parameters.toList))
	logger.finer("post-optimization: %s".format(parameters.toList))
      }

      (0 to data.length - 1).map(oneSample(_))
      optimizeHyperParameters(transitionPriors, historyByTag, historyCounts)
      optimizeHyperParameters(emissionPriors, tagByWord, tagCounts)
      logger.fine("tag counts: %s".format(tagCounts.toList))
      logger.fine("Perplexity: %f".format(perplexity()))
      logger.fine("Best-Match F-Score: %f".format(evaluateOneToOne(data)))
      logger.fine("Variation of Information: %f".format(variationOfInformation()))
    }

    for(i <- 1 until numBurnins + 1){
      logger.fine("burnin #%d".format(i))
      fullSample(dataSet)
      val totals = tagByWord.map(_.fold(0)((x, y) => y + x))
    }
    for(i <- 1 until numSamples + 1){
      logger.fine("sample #%d".format(i))
      fullSample(dataSet)
      val totals = tagByWord.map(_.fold(0)((x, y) => y + x))
    }
    tagByWord.map(printTopic(_))
    def printTopic(row : Seq[Int], count : Int = 10) : Unit = {
      logger.info("%s".format(row.zip(0 to row.size).sorted.reverse.toList.filter(_._1 > 0).map(x => (indexToWord(x._2), x._1, wordCounts(x._2))).take(count)))
    }
  }
}
