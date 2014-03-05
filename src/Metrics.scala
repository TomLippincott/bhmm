package bhmm

import bhmm.Types._
import bhmm.Probability._

object Metrics{

  def evaluateGreedyOneToOne(data : DataSet) : Double = {
    val validLocations = data.filter(_._1 != -1)
    val hypothesisTags = validLocations.map(_._2).toSet
    val goldTags = validLocations.map(_._3).toSet
    val cm = Array.fill[Int](hypothesisTags.size, goldTags.size)(0)
    /*
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
    */
    1.0
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
    /*
    val total = wordCounts.sum
    val hypothesis = data.filter(_._1 != -1).map(_._2)
    val gold = data.filter(_._1 != -1).map(_._3)
    val hypothesisEntropy = entropy(hypothesis)
    val goldEntropy = entropy(gold)
    hypothesisEntropy + goldEntropy - (2 * mutualInformation(hypothesis, gold))
    */
    1.0
  }

}
