import org.apache.commons.math3.special.Gamma
import java.io.File
import Accessories._
import java.io._
import java.util.HashMap
import java.util.Vector
import java.util.Date
import java.util.zip._
import java.util.logging._
import AdaptorGrammar.AdaptorSet
import AdaptorGrammar.RuleSet
import AdaptorGrammar.PitmanYorPrior
import Sampler.AdaptorSampler
import Data.Setting
import Data.AdaptorBasedDataSet

/*
import CFG.NonTerminal
import CFG.Rule
import CFG.RuleRightHand
import CFG.Terminal
import Data._
import Parser.CKY
*/

object BHMM{  

  def time[R](code : => R) : R = {
    val t0 = System.nanoTime()
    val result = code    // call-by-name
    val t1 = System.nanoTime()
    println("Elapsed time: " + (t1 - t0) + "ns")
    result
  }

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

  val base = 2
  def logbase(x: Double) = scala.math.log(x) / scala.math.log(base)
  def expbase(x: Double) = scala.math.pow(base.toDouble, x)

  def fromProb(x : Double) : Prob = new Prob(logbase(x))
  def fromLog(x : Double) : Prob = new Prob(x)

  class Prob(val l : Double) extends Ordered[Prob]{
    def compare(that : Prob) = if(this.l - that.l < 0){ -1 }else if(this.l == that.l){ 0 }else{ 1 }
    def p = expbase(l)
    def *(y : Prob) : Prob = fromLog(l + y.l)
    def /(y : Prob) : Prob = fromLog(l - y.l)
    def inverse = fromProb(1.0) / this
    def +(y : Prob) : Prob = {
      def go(a : Double, b : Double) : Prob = {
	val d = a - b
	if(d < -20){ fromLog(b) }else{ fromLog(b + logbase(1.0 + expbase(d))) }
      }
      if(l < y.l){
	go(l, y.l) 
      }
      else{ 
	go(y.l, l) 
      }
    }
    override def toString = "Prob(%f/%s)".format(p, if(l == Double.NegativeInfinity){ "-" }else{ "%f".format(l) })
  }

  class Dist(vs : Seq[Prob]){
    val scale = new Prob(vs.tail.fold(vs.head)((x, y) => x + y).l).inverse
    def ps = vs.map(x => x * scale)
    def sample(d : Double) : Int = {
      val p = fromProb(d)
      val ms = ps.tail.scanLeft(ps.head)((x, y) => x + y)
      val v = ms.indexWhere((x => p <= x))
      logger.finer("sampled %d: %s %s".format(v, p, ms))
      assert(v < vs.length)
      v
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

  def indexToHistory(i : Int, markov : Int, possibleValues : Int, acc : Seq[Int] = Seq()) : Seq[Int] = {
    i match{
      case 0 =>
	if(acc.length < markov){ indexToHistory(0, markov, possibleValues, 0 +: acc) }else{ acc }
      case _ =>
	val c = scala.math.pow(possibleValues.toDouble, markov - acc.length - 1).toInt
	val d = i / c
	indexToHistory(i - (c * d), markov, possibleValues, d +: acc)
    }
  }
    
  def historyToIndex(history : Seq[Int], possibleValues : Int) : Int = {
    logger.finest("history: %s".format(history))
    val i = history.zipWithIndex.map(x => x._1 * scala.math.pow(possibleValues.toDouble, x._2.toDouble)).sum.toInt
    logger.finest("index: %s".format(i))
    i
  }

  def numHistories(markov : Int, possibleValues : Int) : Int = math.pow(possibleValues.toDouble, markov.toDouble).toInt

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

  type Location = Tuple3[Int, Int, Int]
  type Lookup = scala.collection.mutable.Map[String, Int]
  type VectorCounts = scala.collection.mutable.ArraySeq[Double]
  type Counts = scala.collection.mutable.ArraySeq[VectorCounts]  
  type DataSet = scala.collection.mutable.ArraySeq[Location]

  def parseInput(numLines: Int, numTags : Int, markov: Int, filename: String, wordLookup: Lookup, tagLookup: Lookup) : DataSet = {
    val nullLoc = (-1, numTags, 0)
    val end = (0 to markov - 1).map(_ => nullLoc)
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
    collection.mutable.ArraySeq[Location](src.getLines().take(numLines).map(parseSentence(_)).toList.flatten.++(end).toArray : _*) ++ (0 until markov).map(_ => nullLoc)
  }

  def runTests = {

    val rng = new scala.util.Random()

    // test the history-to-index mechanisms, with 20 tags (20 + 1, including the null tag)
    (1 to 3).map { m =>
      val n = numHistories(m, 20 + 1)
      logger.fine("markov=%d, tags=%d, histories=%d".format(m, 20, n))
      (0 to n - 1).map { i =>
	val h = indexToHistory(i, m, 20 + 1)
	val ii = historyToIndex(h, 21)
	val hh = indexToHistory(ii, m, 20 + 1)
	assert(i == ii && h == hh && h.length == m && hh.length == m)
      }
    }

    // test the log-probability mechanisms
    val xsa = (1 to 4).map(x => rng.nextDouble)
    val psa = xsa.map(_ / xsa.sum)
    val lpsa = ProbSeqToDist(psa.map(fromProb(_)))
    logger.fine("distribution A: %s".format(lpsa))
    val resultsa = Array.fill[Int](4)(0)    
    (1 to 10000).map(x => rng.nextDouble).map(x => resultsa(lpsa.sample(x)) += 1)
    logger.fine("samples A: %s".format(resultsa.toList))

    val xsb = (1 to 4).map(x => rng.nextDouble)
    val psb = xsb.map(_ / xsb.sum)
    val lpsb = ProbSeqToDist(psb.map(fromProb(_)))
    logger.fine("distribution B: %s".format(lpsb))
    val resultsb = Array.fill[Int](4)(0)    
    (1 to 10000).map(x => rng.nextDouble).map(x => resultsb(lpsb.sample(x)) += 1)
    logger.fine("samples B: %s".format(resultsb.toList))

    val lpsab = lpsa * lpsb
    logger.fine("distribution A * B: %s".format(lpsab))
    val lpsba = lpsb * lpsa
    logger.fine("distribution B * A: %s".format(lpsba))
    val resultsab = Array.fill[Int](4)(0)    
    (1 to 10000).map(x => rng.nextDouble).map(x => resultsab(lpsab.sample(x)) += 1)
    logger.fine("samples A * B: %s".format(resultsab.toList))

  }

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

    val prefixPrior = new PitmanYorPrior(0, options.get('prefixPrior).get.asInstanceOf[Double])
    val wordPrior = new PitmanYorPrior(0, options.get('wordPrior).get.asInstanceOf[Double])
    val suffixPrior = new PitmanYorPrior(0, options.get('suffixPrior).get.asInstanceOf[Double])
    val subMorphPrior = new PitmanYorPrior(0, options.get('subMorphPrior).get.asInstanceOf[Double])
    val adaptorPrior = new PitmanYorPrior(0, options.get('adaptPrior).get.asInstanceOf[Double])
    
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

    /*
    val adaptorSet = new AdaptorSet
    val settings = new Setting(prefixPrior, wordPrior, suffixPrior, subMorphPrior, adaptorPrior, ruleDirichletPrior, numTags, prefixes, suffixes, multipleStems, subMorphs, nonParametric)
    val ruleSet = new RuleSet(settings)
    val sampler = new AdaptorSampler(settings)
    val adaptorBasedDataSet = new AdaptorBasedDataSet
    adaptorBasedDataSet.initialSegmentation(settings, ruleSet, adaptorSet)
    val writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(outputPath), "utf-8"))    
    val writer2 = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(outputPath2), "utf-8"))      
    AdaptorBasedDataSet adaptorBasedDataSet = TypeBasedDataReader.readAdaptorDataSet(inputPath)
    */

    var transitionPriors = (0 to numTags).map(_ => options.get('transitionPrior).get.asInstanceOf[Double]).toArray
    var emissionPriors = (0 to numWords - 1).map(_ => options.get('emissionPrior).get.asInstanceOf[Double]).toArray

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

      def entropy(as : Seq[Int]) : Double = {
	val total = as.length
	val cts = as.distinct.map(x => as.count(y => y == x))
	val dist = new Dist(cts.map(t => fromProb(t.toDouble / total)))
	-dist.ps.filter(x => x.l != Double.NaN && x.l != Double.NegativeInfinity).map(x => x.p * x.l).sum
      }

      def mutualInformation(aas : Seq[Int], bas : Seq[Int]) : Double = {
	val total = aas.length
	val combined = aas.zip(bas)
	assert(total == bas.length)
	val aCounts = aas.distinct.map(x => (x, aas.count(y => y == x))).toMap
	val bCounts = bas.distinct.map(x => (x, bas.count(y => y == x))).toMap
	val abCounts = aCounts.keys.map(a => bCounts.keys.map(b => ((a, b), combined.count(_ == (a, b))))).flatten.toMap
	val abProb = abCounts.mapValues(_ / total)
	val aProb = aCounts.mapValues(_ / total)
	val bProb = bCounts.mapValues(_ / total)
	aProb.keys.filter(a => aProb(a) > 0.0).map(a => bProb.keys.filter(b => bProb(b) > 0.0).map{b =>
	  abProb((a, b)) * logbase(abProb((a, b)) / (aProb(a) * bProb(b)))
					 }).flatten.sum
      }

      def variationOfInformation() : Double = {
	val total = wordCounts.sum
	val hypothesis = data.filter(_._1 != -1).map(_._2)
	val gold = data.filter(_._1 != -1).map(_._3)
	val hypothesisEntropy = entropy(hypothesis)
	val goldEntropy = entropy(gold)
	hypothesisEntropy + goldEntropy - (2 * mutualInformation(hypothesis, gold))
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
      
      def symmetricOptimizeHyperParameters(parameters : Array[Double], groupByObsCounts : Array[Array[Int]], iterations : Int) : Unit = {
	val currentValue = parameters.sum / parameters.length
	val numGroups = groupByObsCounts.length
	val numObs = parameters.length	
	val groupCounts = groupByObsCounts.map(_.sum)
	assert(numObs == groupByObsCounts(0).length)
	val numeratorA = (0 to numObs - 1).map(o => (0 to numGroups - 1).map(g => Gamma.digamma(groupByObsCounts(g)(o) + currentValue))).flatten.sum
	val numeratorB = numGroups * numObs * Gamma.digamma(currentValue)
	val numerator = currentValue * (numeratorA - numeratorB)
	val denominatorA = groupCounts.map(x => Gamma.digamma(x + numObs * currentValue)).sum
	val denominatorB = numGroups * Gamma.digamma(numObs * currentValue)
	val denominator = numObs * (denominatorA - denominatorB)
	val newValue = currentValue * (numerator / denominator) + .00001
	(0 to numObs - 1).map(o => parameters(o) = newValue)
	if(iterations <= 0){ Unit }else{ symmetricOptimizeHyperParameters(parameters, groupByObsCounts, iterations - 1) }
      }

      def optimizeHyperParameters(parameters : Array[Double], groupByObsCounts : Array[Array[Int]], iterations : Int) : Unit = {
	val numGroups = groupByObsCounts.length
	val numObs = parameters.length
	val currentSum = parameters.sum
	val numerators = (0 to numObs - 1).map(o => groupByObsCounts.map(n => Gamma.digamma(n(o) + parameters(o))).sum - (numGroups * Gamma.digamma(parameters(o))))
	val denominator = groupByObsCounts.map(n => Gamma.digamma(n.sum + currentSum)).sum - (numGroups * Gamma.digamma(currentSum))
	val update = numerators.zipWithIndex.map(x => parameters(x._2) * x._1 / denominator)
	(0 to numObs - 1).map(o => parameters(o) = update(o) + .00001)
	if(iterations <= 0){ logger.finer("post-optimization: %s".format(parameters.toList)) }else{ optimizeHyperParameters(parameters, groupByObsCounts, iterations - 1) }
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
      if(printPerplexity > 0 && number % printPerplexity == 0) logger.fine("Perplexity: %f".format(perplexity()))
      if(printBestMatch > 0 && number % printBestMatch == 0) logger.fine("Best-Match F-Score: %f".format(evaluateOneToOne(data)))
      if(printVariationOfInformation > 0 && number % printVariationOfInformation == 0) logger.fine("Variation of Information: %f".format(variationOfInformation()))
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
