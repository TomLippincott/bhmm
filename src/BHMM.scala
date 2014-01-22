import java.io.File

import Accessories._

import java.io._
import java.util.HashMap
import java.util.Vector

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


  def parseInput(numLines: Int, markov: Int, filename: String, wordLookup: Lookup, tagLookup: Lookup) : DataSet = {
    val nullLoc = Tuple3[Int, Int, Int](-1, 0, 0)
    def parseLocation(loc: String): Location = {
      val toks = loc.split("/")
      val word = toks(0)
      val tag = toks(1)      
      val wordId = if(wordLookup.contains(word)){ wordLookup(word) }else{ wordLookup += Tuple2(word, wordLookup.size); wordLookup(word) }
      val goldTagId = if(tagLookup.contains(tag)){ tagLookup(tag) }else{ tagLookup += Tuple2(tag, tagLookup.size); tagLookup(tag) }
      val tagId = -1
      (wordId, tagId, goldTagId)
    }
    def parseSentence(line: String): Seq[Location] = {
      (0 until markov).map(_ => nullLoc) ++ line.split(" ").map(parseLocation(_))
    }
    collection.mutable.ArraySeq[Location](scala.io.Source.fromFile(new File(filename)).getLines().take(numLines).map(parseSentence(_)).toList.flatten.toArray : _*)
  }

  def sample(target: Double, dist: Seq[Double]) : Int = {
    val summed = dist.map(math.exp(_)).foldLeft[Seq[Double]](Seq(0.0))((a, b) => a :+ (b + a.last)).drop(1)
    val scaled = target * summed.last
    val ret = summed.indexWhere(scaled < _)
    ret
  }

  def windows(size : Int, total : Int) : Seq[Seq[Int]] = {
    (0 to total - size).map(x => x to x + size)
  }

  def sadegh(config: String) = {
    val setting = SettingReader.getSetting(config) //"Test/setting.txt")
    val ruleset = new RuleSet(setting)
    val adaptorSet = new AdaptorSet()
    /*
    Setting setting = SettingReader.getSetting(offset+"Test/setting.txt");
    RuleSet ruleSet=new RuleSet(setting);
    AdaptorSet adaptorSet=new AdaptorSet();
    
    int iteration=100;
      
	   String inputPath= offset+"Test/train.verb";
      String devPath=offset+"Test/dev.verb";
      String outputPath=offset+"Test/tmp.txt";
      String outputPath2=offset+"Test/tmp2.txt";
      if(args.length>1){
      inputPath=args[0];
      outputPath=args[1];
            outputPath2=args[2];
            iteration=Integer.parseInt(args[3]) ;
        }

        WordBasedDataSet wordBasedDataSet= TypeBasedDataReader.readDataSet(inputPath);
        wordBasedDataSet.initialSegmentation(setting,ruleSet,adaptorSet);

        BufferedWriter writer = new BufferedWriter
                (new OutputStreamWriter(new FileOutputStream(outputPath),"utf-8"));
        BufferedWriter writer2 = new BufferedWriter
                (new OutputStreamWriter(new FileOutputStream(outputPath2),"utf-8"));
        AdaptorSampler sampler=new AdaptorSampler(setting);
	*/
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
	case "--help" :: tail =>
	  println(usage)
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
				 'emissionPrior -> .1),arglist)
    val config = options.get('config).get.asInstanceOf[String]
    val input = options.get('input).get.asInstanceOf[String]
    val output = options.get('input).get.asInstanceOf[String]
    val markov = options.get('markov).get.asInstanceOf[Int]
    val numSentences = options.get('numSentences).get.asInstanceOf[Int]
    val numTags = options.get('numTags).get.asInstanceOf[Int] + 1
    val numBurnins = options.get('numBurnins).get.asInstanceOf[Int]
    val numSamples = options.get('numSamples).get.asInstanceOf[Int]
    
    val tagLookup = scala.collection.mutable.Map[String, Int]()
    val wordLookup = scala.collection.mutable.Map[String, Int]()
    
    val dataSet = parseInput(numSentences, markov, input, wordLookup, tagLookup)
    
    val numWords = wordLookup.size
    val numHistories = scala.math.pow((numTags.toDouble + 1), markov.toDouble).toInt
    
    val tagByWord = Array.fill[Int](numTags - 1, wordLookup.size)(0)
    val historyByTag = Array.fill[Int](numHistories, numTags - 1)(0)
    val historyCounts = Array.fill[Int](numHistories)(0)
    val tagCounts = Array.fill[Int](numTags)(0)

    val transitionPriors = (0 to numTags).map(_ => options.get('transitionPrior).get.asInstanceOf[Double])
    val emissionPriors = (0 to numWords).map(_ => options.get('emissionPrior).get.asInstanceOf[Double])

    //sadegh(config)
    
    val rng = new scala.util.Random()
    
    def indexToHistory(t : Int, m : Int, i : Int, acc : Seq[Int] = Seq()) : Seq[Int] = {
      m match{
	case 0 => 
	  acc
	case _ =>
	  //c = t ^ (m' - 1)
          //v = fromMaybe t (findIndex (\x -> (x * c) > i') [0..t]) - 1
	  val c = math.pow(t, m - 1)
	  val v = 1
	  acc
	//indexToHistory(m - 1, i - (v * c), v +: acc)
      }
    }
    
    def historyToIndex(history: Seq[Int]) : Int = {
      history.zip(0 to history.size).map(x => x._2 * scala.math.pow(numTags.toDouble, x._1.toDouble)).fold(0.0)((x, y) => x + y).toInt
    }
    
    def fullSample(data : DataSet) : Unit = {
      def crement(index : Int, amount : Int) : Unit = {
	val (wordId, tagId, goldTagId) = data(index)
	tagByWord(tagId)(wordId) += amount
	tagCounts(tagId) += amount	
      }
      def oneSample(index : Int) : Unit = {
	val loc = data(index)
	data(index) match{
	  case (-1, _, _) =>
	    Unit
	  case (wordId, tagId, goldTagId) =>
	    if(tagId != -1){
	      crement(index, -1)
	    }
	    val transitionDists = (0 to markov).map(_ => (0 to numTags - 1).map(_ => 1.0).map(scala.math.log(_)))
	    val emissionDist = tagByWord.map(_(wordId).toDouble + emissionPriors(wordId)) //.map(scala.math.log(_))
	    val dist = transitionDists.zip(emissionDist).map(x => x._2)
	    val newTag = sample(rng.nextDouble, dist)
	    data(index) = (wordId, newTag, goldTagId)
	    crement(index, 1)			
	}
      }
      (markov to data.length - 1).map(oneSample(_))
    }
    
    for(i <- 1 until numBurnins + 1){
      println("burnin #" + i.toString)
      fullSample(dataSet)
      val totals = tagByWord.map(_.fold(0)((x, y) => y + x))
      println(totals.toList)
    }
    
    for(i <- 1 until numSamples + 1){
      println("sample #" + i.toString)
      fullSample(dataSet)
      val totals = tagByWord.map(_.fold(0)((x, y) => y + x))
      println(totals.toList)
    }
    val rWordLookup = wordLookup.map(_.swap)
    tagByWord.map(printTopic(_))
    println(rWordLookup.size)
    def printTopic(row : Seq[Int], count : Int = 10) : Unit = {
      println(row.zip(0 to row.size).sorted.reverse.toList.filter(_._1 > 0).map(x => (rWordLookup(x._2), x._1)).take(count))
    }
  }
}
