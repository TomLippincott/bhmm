package bhmm

import java.util.zip._
import java.io.File
import java.io._
import java.util.logging._

object Utilities{
  val logger = Logger.getLogger("bhmm")
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
    val i = history.zipWithIndex.map(x => x._1 * scala.math.pow(possibleValues.toDouble, x._2.toDouble)).sum.toInt
    i
  }

  def numHistories(markov : Int, possibleValues : Int) : Int = math.pow(possibleValues.toDouble, markov.toDouble).toInt
}
