package bhmm.types

class Probability(val v : Double) extends Numeric[Probability] with Ordered[Probability]{
  def p = Probability.expbase(v)
  def compare(x : Probability, y : Probability) : Int = x.v.compare(y.v)
  def compare(y : Probability) : Int = v.compare(y.v)
  def minus(x : Probability, y : Probability) : Probability = sys.error("tried to subtract Probabilities")
  def plus(x : Probability, y : Probability) : Probability = {
    def go(s : Double, l : Double) : Probability = {
      val d = s - l
      if(d < -20){ Probability.fromLog(l) }else{ Probability.fromLog(l + Probability.logbase(1.0 + Probability.expbase(d))) }
    }
    if(x < y){
      go(x.v, y.v) 
    }
    else{ 
      go(y.v, x.v) 
    }
  }
  def +(y : Probability) : Probability = plus(this, y)
  def *(y : Probability) : Probability = times(this, y)
  def /(y : Probability) : Probability = divide(this, y)
  def ^(e : Double) : Probability = Probability.fromLog(e * v)
  def inverse = Probability.fromProb(1.0) / this
  def times(x : Probability, y : Probability) : Probability = Probability.fromLog(x.v + y.v)
  def divide(x : Probability, y : Probability) : Probability = Probability.fromLog(x.v - y.v)
  def fromInt(i : Int) : Probability = sys.error("tried to convert an Int to a Probability")
  def toDouble(x : Probability) : Double = sys.error("tried to convert a Probability to a Double")
  def toFloat(x : Probability) : Float = sys.error("tried to convert a Probability to a Float")
  def toInt(x : Probability) : Int = sys.error("tried to convert a Probability to an Int")
  def toLong(x : Probability) : Long = sys.error("tried to convert a Probability to a Long")
  def negate(x : Probability) : Probability = sys.error("")
  override def toString = "Prob(%f/%s)".format(p, if(v == Double.NegativeInfinity){ "-" }else{ "%f".format(v) })
}

object Probability{
  val base = 2
  def logbase(x: Double) = scala.math.log(x) / scala.math.log(base)
  def expbase(x: Double) = scala.math.pow(base.toDouble, x)
  def fromProb(x : Double) : Probability = new Probability(logbase(x))
  def fromLog(x : Double) : Probability = new Probability(x)
}
