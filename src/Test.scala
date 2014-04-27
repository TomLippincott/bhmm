package bhmm

import java.util.logging._

object Test{
  val logger = Logger.getLogger(Test.getClass.getName)
  def runTests = {
  /*
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
  */
  }

}
