package bhmm

object Types{
  type Location = Tuple3[Int, Int, Int]
  type Lookup = scala.collection.mutable.Map[String, Int]
  type VectorCounts = scala.collection.mutable.ArraySeq[Double]
  type Counts = scala.collection.mutable.ArraySeq[VectorCounts]  
  type DataSet = scala.collection.mutable.ArraySeq[Location]
  
}
