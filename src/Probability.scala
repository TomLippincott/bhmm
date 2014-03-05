package bhmm

object Probability{
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
      //logger.finer("sampled %d: %s %s".format(v, p, ms))
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
}
