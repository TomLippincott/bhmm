package bhmm

import org.apache.commons.math3.special.Gamma
import java.util.logging._

object Dirichlet{
  val logger = Logger.getLogger(Dirichlet.getClass.getName)
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

  def asymmetricOptimizeHyperParameters(parameters : Array[Double], groupByObsCounts : Array[Array[Int]], iterations : Int) : Unit = {
    val numGroups = groupByObsCounts.length
    val numObs = parameters.length
    val currentSum = parameters.sum
    val numerators = (0 to numObs - 1).map(o => groupByObsCounts.map(n => Gamma.digamma(n(o) + parameters(o))).sum - (numGroups * Gamma.digamma(parameters(o))))
    val denominator = groupByObsCounts.map(n => Gamma.digamma(n.sum + currentSum)).sum - (numGroups * Gamma.digamma(currentSum))
    val update = numerators.zipWithIndex.map(x => parameters(x._2) * x._1 / denominator)
    (0 to numObs - 1).map(o => parameters(o) = update(o) + .00001)
    if(iterations <= 0){ Unit }else{ asymmetricOptimizeHyperParameters(parameters, groupByObsCounts, iterations - 1) }
  }

}
