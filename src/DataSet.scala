package bhmm.data

import bhmm.Utilities._
import java.util.zip._
import java.io.File
import java.io._
import java.util.logging._
import scala.collection.mutable
import scala.xml.XML
import scala.io.Source
import scala.xml
import scala.xml.dtd.{DocType,SystemID}

class DataSet(val sentences : Seq[Seq[(Int, Option[Int], Seq[Int])]],
	      val indexToWord : Map[Int, String],
	      val indexToTag : Map[Int, String],
	      val indexToAnalysis : Map[Int, Seq[String]]) extends Seq[Seq[(Int, Option[Int], Seq[Int])]]{
  

  val locations = (0 until sentences.length).map(s => (0 until sentences(s).length).map(w => (s, w))).flatten
  val wordToIndex = indexToWord.map(_.swap)
  val tagToIndex = indexToTag.map(_.swap)
  val analysisToIndex = indexToAnalysis.map(_.swap)
  lazy val words : Seq[String] = wordToIndex.keys.toList
  override def toString() : String = "DataSet: %d sentences, %d tokens, %d word types, %d analyses, %d tags".format(sentences.length, sentences.map(_.length).sum, wordToIndex.size, analysisToIndex.size, tagToIndex.size)
  def print() : String = sentences.map{ s => s.map{ l => "%s/%s".format(indexToWord(l._1), 
									indexToTag.getOrElse(l._2.getOrElse(-1), "-")
								      )}.mkString(" ")}.mkString("\n")

  def apply(i : Int) : Seq[(Int, Option[Int], Seq[Int])] = sentences(i)
  def length : Int = sentences.length
  def iterator : Iterator[Seq[(Int, Option[Int], Seq[Int])]] = sentences.iterator
  def save(out : Writer) : Unit = {
    val top = 
    <dataset>
      <preamble>
	<analysis_inventory>
	{
	  analysisToIndex.toSeq.sortBy(_._2).map{ case (a, i) => <entry id={ i.toString }>{ 
	    a.map{
	      m =>
		val toks = m.split("_")
		if(List("PRE", "STM", "SUF").map(toks.last.startsWith(_)).contains(true)){ 
		  <morph type={ toks.last }>{ toks.dropRight(1).mkString("_") }</morph>
		}else{ 
		  <morph>{ m }</morph>
		}
	    }	    
	  }</entry>}
	}
	</analysis_inventory>
	<tag_inventory>
	{
	  tagToIndex.toSeq.sortBy(_._2).map{ case (t, i) => <entry id={ i.toString }>{ t }</entry>}
	}
	</tag_inventory>
	<word_inventory>
	{
	  wordToIndex.toSeq.sortBy(_._2).map{ case (w, i) => <entry id={ i.toString }>{ w }</entry>}
	}
	</word_inventory>
      </preamble>
	<sentences>
	{
	  sentences.map{
	    s =>
	      <sentence>
	      {
		s.map{
		  case (wordId, Some(tagId), analysisIds) =>
		    <location>
		      <word id={ wordId.toString } />
		      <tag id={ tagId.toString } />
		      <analyses>
		      {
			analysisIds.map(a => <analysis id={ a.toString } />)
		      }
		      </analyses>
		    </location>
		  case (wordId, None, analysisIds) =>
		    <location>
		      <word id={ wordId.toString } />
		      <analyses>
		      {
			analysisIds.map(a => <analysis id={ a.toString } />)
		      }
		      </analyses>
		    </location>
		}
	      }
	      </sentence>
	  }
	}
	</sentences>
    </dataset>
    XML.write(out, top, "utf-8", false, null)
  }   
}

object DataSet{
  type Location = (Int, Option[Int], Seq[Int])
  type Sentence = Seq[Location]
  type Sentences = Seq[Sentence]

  def fromFile(fileName : String, tagMap : Map[String, String] = Map()) : DataSet = {
    val src = if(fileName.endsWith("gz")){ new GZIPInputStream(new BufferedInputStream(new FileInputStream(fileName))) }
	      else{ new BufferedInputStream(new FileInputStream(fileName)) }
    val (sentences, words, tags, analyses) = if(fileName.endsWith("xml") || fileName.endsWith("xml.gz")){ fromXML(src, tagMap) }else{ fromXML(src, tagMap) }
    new DataSet(sentences, words, tags, analyses)
  }
  
  abstract trait XmlLoc
  case class Outside() extends XmlLoc
  case class InTag() extends XmlLoc
  case class InMorph() extends XmlLoc
  
  def fromXML(src : InputStream, tagMap : Map[String, String] = Map(), trainOnly : Boolean=true) : (Sentences, Map[Int, String], Map[Int, String], Map[Int, Seq[String]]) = {
    val xml = XML.load(src)
    val words = (xml \ "dataset" \ "preamble" \ "word_inventory" \ "entry").map(w => ((w \ "@id").text.toInt, w.text)).toMap
    val tags = (xml \ "dataset" \ "preamble" \ "tag_inventory" \ "entry").map(t => ((t \ "@id").text.toInt, t.text)).toMap
    val analyses = (xml \ "dataset" \ "preamble" \ "analysis_inventory" \ "entry").map(a => ((a \ "@id").text.toInt, (a \ "morph").map(_.text))).toMap
    val sentences = (xml \\ "sentence").map(s => (s \ "location").map{
      l => 
	val wordId = (l \ "word" \ "@id").text.toInt
	val tagIds = (l \ "tag" \ "@id").map(_.text.toInt)
	val tagId = if(tagIds.size == 0){ None }else{ Some(tagIds(0)) }
	val analysisIds = (l \ "analyses" \ "analysis" \ "@id").map(_.text.toInt)
	(wordId, tagId, analysisIds)
    })
    (sentences, words, tags, analyses)
  }

  def toLowerCase(data : DataSet) : DataSet = {
    val oldIndexToOldWord = data.indexToWord
    val oldIndexToNewWord = oldIndexToOldWord.mapValues(_.toLowerCase)
    val wordToIndex = oldIndexToNewWord.values.toSet.zipWithIndex.toMap
    val indexToWord = wordToIndex.map(_.swap)
    val sentences = data.sentences.map{
      s =>
	s.map{ l => (wordToIndex(oldIndexToNewWord(l._1)), l._2, l._3) }
    }
    new DataSet(sentences, indexToWord, data.indexToTag, data.indexToAnalysis)
  }
}
