package bhmm

import java.util.zip._
import java.io.File
import java.io._
import java.util.logging._
import AdaptorGrammar.Adaptor
import Data.Segmentation
import java.lang.String
import bhmm.types.Probability
import bhmm.types.Probability.{fromProb}
import bhmm.types.Distribution
import bhmm.types.Distribution.{fromProbs}

object Utilities{
  val logger = Logger.getLogger("bhmm")

  implicit def adaptorToString(a : AdaptorGrammar.Adaptor) : String = {
    val w = a.getWord()
    val s = new Segmentation(w, 1)
    s.completeSegmentationSpans(a, 0, true)
    s.toSegmentedString(true).split(" ").map(_.split("_")(0)).mkString("+")
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

  def forward(markov : Int, observations : Seq[Int], _transition : Array[Array[Double]], _emission : Array[Array[Double]]) : Seq[Probability] = {

    // state to state
    val numObservations = _emission(0).size
    val numStates = _transition(0).size
    val newNumStates = numHistories(markov, numStates)
    /*
    println(numObservations)
    println(numStates)
    println(newNumStates)
    */
    val transition = (0 until newNumStates).map{
      fromIndex =>
	val fromHist = indexToHistory(fromIndex, markov, numStates)
	(0 until newNumStates).map{
	  toIndex =>
	    val toHist = indexToHistory(toIndex, markov, numStates)
	    if(fromHist.drop(1) == toHist.dropRight(1)){ _transition(fromHist.last)(toHist.last) }else{ 0.0 }
	}
    }

    // state to observation
    val emission = (0 until newNumStates).map{
      fromIndex =>
	val lastState = indexToHistory(fromIndex, markov, numStates).last
	_emission(lastState)

    }

    var alphaStateProbs = (0 until transition.size).map(_ => 1.0) //indexToHistory(_, markov, numStates)).filter(_.last < numStates).map((_, fromProb(1.0))).toMap
    observations.map{
      obs =>
	alphaStateProbs = if(obs == -1){ alphaStateProbs }else{
	  //val state = states(i)
	  //val hist = histories(i)
	  //val eprob = emission(state)(obs)
	  alphaStateProbs.zipWithIndex.map{
	    case (prob, history) =>
	      val state = indexToHistory(history, markov, numStates).last
	      val emProb = if(state == numStates){ 1.0 }else{ emission(state)(obs) }
	      emProb * prob
	  }
	  /*(0 until alpha.size).map{
	    h =>
	      val states = indexToHistory(h, markov, numStates)
	      val state = states.last
	      emission(states.last)(obs)*/
	    //eprob
	  //}
	  //println((obs, state, hist, eprob, numStates))
	  //alpha
	}
    }
    alphaStateProbs.map(fromProb(_)) //.values.toSeq
    //Seq(Probability.fromProb(alpha.sum))
  }

}
