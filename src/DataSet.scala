package bhmm

import bhmm.Types._
import bhmm.Utilities._
import java.util.zip._
import java.io.File
import java.io._

object DataSet{
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
}
